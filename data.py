"""Data loading for the CoLLM ML-1M benchmark.

Produces two views of the same data:
  (a) per-sample (user, target_item, label, history) records for both SASRec CTR
      pre-training and the LLM stage, and
  (b) a grouped-by-user batch sampler so within-user (pos, neg) pairs exist in a
      batch for the pairwise UAUC-targeting loss.

Item-id convention: CoLLM's ML-1M pickles are ALREADY 1-based with id 0 reserved
as a "no history" placeholder (its title is the empty string), so we use 0 directly
as the padding id whose embedding is frozen at zero in SASRec. Verified on the real
files: uid in 1..838, iid in 1..3255, and 0 appears only as a leading placeholder
inside `his`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

PAD_ID = 0  # internal padding item id; embedding frozen at zero


# --------------------------------------------------------------------------- #
# Loading CoLLM's preprocessed pickles
# --------------------------------------------------------------------------- #

def _to_list(x):
    """CoLLM stores history as list / np.ndarray depending on pandas version."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def load_collm_ml1m(data_dir: str | Path):
    """Load CoLLM's train/valid/test _ood2 pickles.

    Returns (train_df, val_df, test_df, n_users, n_items, id2title) where
    dataframes have normalized columns: uid, iid, his (list of real item ids,
    0-placeholders stripped), label (float), and id2title maps item id -> title.
    """
    data_dir = Path(data_dir)
    dfs = {}
    for split, fname in [("train", "train_ood2.pkl"),
                         ("val", "valid_ood2.pkl"),
                         ("test", "test_ood2.pkl")]:
        with open(data_dir / fname, "rb") as f:
            dfs[split] = pickle.load(f)

    id2title: dict[int, str] = {}

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        out["uid"] = df["uid"].astype(int).values
        out["iid"] = df["iid"].astype(int).values
        # CoLLM marks "no history yet" with a single leading id 0 -> strip it
        out["his"] = [[int(i) for i in _to_list(h) if int(i) != PAD_ID] for h in df["his"]]
        out["label"] = df["label"].astype(float).values
        # harvest titles from both target and history title columns
        for iid, t in zip(out["iid"], df["title"]):
            id2title.setdefault(int(iid), str(t))
        for h_raw, hts in zip(df["his"], df["his_title"]):
            for iid, t in zip(_to_list(h_raw), _to_list(hts)):
                if int(iid) != PAD_ID:
                    id2title.setdefault(int(iid), str(t))
        return out

    train_df = normalize(dfs["train"])
    val_df = normalize(dfs["val"])
    test_df = normalize(dfs["test"])

    all_df = pd.concat([train_df, val_df, test_df])
    n_users = int(all_df["uid"].max()) + 1
    max_item = max(int(all_df["iid"].max()),
                   max((max(h) for h in all_df["his"] if len(h)), default=0))
    n_items = max_item  # internal ids run 1..n_items; 0 is pad

    return train_df, val_df, test_df, n_users, n_items, id2title


# --------------------------------------------------------------------------- #
# Synthetic data with the same schema (for --smoke_test)
# --------------------------------------------------------------------------- #

def make_synthetic(n_users=60, n_items=120, n_train=1500, n_val=400, n_test=400,
                   max_his=10, dim=8, seed=0):
    """Latent-factor synthetic data so the pipeline has learnable signal.

    Labels come from a user x item dot-product plus noise, thresholded at the
    per-dataset median -> roughly balanced classes and a real (learnable) ranking
    structure, which lets the smoke test sanity-check that UAUC moves above 0.5.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, dim))
    V = rng.normal(size=(n_items + 1, dim))  # internal ids 1..n_items

    def sample_split(n):
        uid = rng.integers(0, n_users, size=n)
        iid = rng.integers(1, n_items + 1, size=n)
        score = (U[uid] * V[iid]).sum(-1) + rng.normal(scale=0.5, size=n)
        label = (score > np.median(score)).astype(float)
        his = []
        for u in uid:
            k = int(rng.integers(3, max_his + 1))
            # bias history toward the user's high-affinity items (positive history)
            logits = V[1:] @ U[u]
            p = np.exp(logits - logits.max()); p /= p.sum()
            his.append((rng.choice(n_items, size=k, replace=False, p=p) + 1).tolist())
        return pd.DataFrame({"uid": uid, "iid": iid, "his": his, "label": label})

    id2title = {i: f"Synthetic Movie #{i} ({1970 + i % 50})" for i in range(1, n_items + 1)}
    return (sample_split(n_train), sample_split(n_val), sample_split(n_test),
            n_users, n_items, id2title)


# --------------------------------------------------------------------------- #
# Dataset / collation
# --------------------------------------------------------------------------- #

@dataclass
class RecBatch:
    uid: torch.Tensor          # [B]
    iid: torch.Tensor          # [B] internal target item ids
    label: torch.Tensor        # [B] float
    his: torch.Tensor          # [B, L] internal ids, LEFT-padded with PAD_ID
    his_mask: torch.Tensor     # [B, L] 1 where real item
    his_titles: list           # list[B] of list[str]
    target_titles: list        # list[B] of str

    def to(self, device):
        self.uid = self.uid.to(device)
        self.iid = self.iid.to(device)
        self.label = self.label.to(device)
        self.his = self.his.to(device)
        self.his_mask = self.his_mask.to(device)
        return self


class RecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id2title: dict[int, str], max_his: int = 10):
        self.uid = df["uid"].to_numpy()
        self.iid = df["iid"].to_numpy()
        self.label = df["label"].to_numpy(dtype=np.float32)
        self.max_his = max_his
        # cap history to the MOST RECENT max_his items (CoLLM order: oldest -> newest)
        self.his = [h[-max_his:] for h in df["his"]]
        self.id2title = id2title

    def __len__(self):
        return len(self.uid)

    def title(self, iid: int) -> str:
        return self.id2title.get(int(iid), f"item {int(iid)}")

    def __getitem__(self, idx):
        his = self.his[idx]
        pad = self.max_his - len(his)
        return {
            "uid": int(self.uid[idx]),
            "iid": int(self.iid[idx]),
            "label": float(self.label[idx]),
            "his": [PAD_ID] * pad + list(his),      # LEFT pad
            "his_mask": [0] * pad + [1] * len(his),
            "his_titles": [self.title(i) for i in his],
            "target_title": self.title(self.iid[idx]),
        }


def collate(batch: list[dict]) -> RecBatch:
    return RecBatch(
        uid=torch.tensor([b["uid"] for b in batch], dtype=torch.long),
        iid=torch.tensor([b["iid"] for b in batch], dtype=torch.long),
        label=torch.tensor([b["label"] for b in batch], dtype=torch.float),
        his=torch.tensor([b["his"] for b in batch], dtype=torch.long),
        his_mask=torch.tensor([b["his_mask"] for b in batch], dtype=torch.float),
        his_titles=[b["his_titles"] for b in batch],
        target_titles=[b["target_title"] for b in batch],
    )


class UserGroupedBatchSampler(Sampler):
    """Batches composed of a few users' samples each, so same-user (pos, neg)
    pairs exist within a batch — required by the pairwise UAUC loss.

    Shuffles users each epoch, then chunks each user's sample indices; a batch
    concatenates chunks from `users_per_batch` users up to `batch_size` samples.
    """

    def __init__(self, uids: np.ndarray, batch_size: int, users_per_batch: int, seed: int = 0):
        self.batch_size = batch_size
        self.users_per_batch = max(1, users_per_batch)
        self.per_user = int(np.ceil(batch_size / self.users_per_batch))
        self.user_to_idx = {}
        for i, u in enumerate(uids):
            self.user_to_idx.setdefault(int(u), []).append(i)
        self.rng = np.random.default_rng(seed)
        self.n_samples = len(uids)

    def __iter__(self):
        # build per-user chunks, shuffled
        chunks = []
        for u, idxs in self.user_to_idx.items():
            idxs = list(idxs)
            self.rng.shuffle(idxs)
            for s in range(0, len(idxs), self.per_user):
                chunks.append(idxs[s:s + self.per_user])
        self.rng.shuffle(chunks)
        batch = []
        for c in chunks:
            batch.extend(c)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))


def load_data(cfg):
    """Entry point: returns (train_ds, val_ds, test_ds, n_users, n_items, id2title)."""
    if cfg.smoke_test:
        tr, va, te, n_users, n_items, id2title = make_synthetic(seed=cfg.seed)
    else:
        data_dir = Path(cfg.data_dir)
        if not (data_dir / "train_ood2.pkl").exists():
            print(f"[data] {data_dir}/train_ood2.pkl not found -> falling back to synthetic data")
            tr, va, te, n_users, n_items, id2title = make_synthetic(seed=cfg.seed)
        else:
            tr, va, te, n_users, n_items, id2title = load_collm_ml1m(data_dir)
    mk = lambda df: RecDataset(df, id2title, max_his=cfg.max_his_len)
    return mk(tr), mk(va), mk(te), n_users, n_items, id2title
