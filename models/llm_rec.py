"""LLM wrapper: frozen backbone + LoRA, with soft-token ("hybrid") encoding.

Prompt (CoLLM's ML-1M template, filled exactly):

  #Question: A user has given high ratings to the following movies:
  <HisItemTitleList>. Additionally, we have information about the user's
  preferences encoded in the feature <UserID>. Using all available information,
  make a prediction about whether the user would enjoy the movie titled
  <TargetItemTitle> with the feature <TargetItemID>? Answer with "Yes" or "No".
  \n#Answer:

<UserID> is N QFormer soft tokens; <TargetItemID> is 1 projected item token.

HYBRID ENCODING: we never invent placeholder vocabulary ids. The template is
split into text segments around the two ID slots; each segment is tokenized and
embedded with the backbone's (frozen) input embedding table, and the soft-token
tensors are spliced in between at the embedding level. The whole batch is then
LEFT-padded so the final position of every row is the token right before the
answer — P("Yes") is read from the next-token logits at that position. Gradients
flow through inputs_embeds back into the QFormer even when the LLM and LoRA are
completely frozen (Phase 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Template split around the two soft-token slots (text is otherwise verbatim).
SEG_A = ("#Question: A user has given high ratings to the following movies: "
         "{his}. Additionally, we have information about the user's preferences "
         "encoded in the feature ")
SEG_B = (". Using all available information, make a prediction about whether the "
         "user would enjoy the movie titled {target} with the feature ")
SEG_C = "? Answer with \"Yes\" or \"No\". \n#Answer:"

# SeLLa-Rec's <Warm_ID> slot: inserted between the item token and SEG_C only
# when warm tokens are passed, so the non-warm template stays byte-identical.
SEG_W = " and the semantic feature "

# Phase-1 text-only variant: same task phrasing with the ID clauses dropped.
TEXT_ONLY = ("#Question: A user has given high ratings to the following movies: "
             "{his}. Using all available information, make a prediction about "
             "whether the user would enjoy the movie titled {target}? "
             "Answer with \"Yes\" or \"No\". \n#Answer:")


def fill_titles(his_titles: list[str], target_title: str, hybrid: bool):
    his = ", ".join(f'"{t}"' for t in his_titles) if his_titles else '"unknown"'
    tgt = f'"{target_title}"'
    if hybrid:
        return SEG_A.format(his=his), SEG_B.format(target=tgt), SEG_C
    return TEXT_ONLY.format(his=his, target=tgt)


class LLMRec(nn.Module):
    def __init__(self, backbone: str = "lmsys/vicuna-7b-v1.5",
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 lora_targets=("q_proj", "v_proj"), load_4bit: bool = False,
                 device: str = "cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {}
        # T4/V100 (Colab free tier) have no bf16 support — fall back to fp16 there
        half = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=half)
            kwargs["device_map"] = {"": device}  # 4-bit models can't be .to()-moved
        elif device == "cuda":
            kwargs["torch_dtype"] = half
        # Repos without main-branch safetensors (e.g. lmsys/vicuna-7b-v1.5, which
        # ships only .bin) make from_pretrained fetch an auto-converted safetensors
        # copy from a Hub conversion PR — a 13.5GB duplicate that defeats any
        # pre-download of the main branch. Prefer the repo's own weights instead.
        try:
            from huggingface_hub import list_repo_files
            if not any(f.endswith(".safetensors") for f in list_repo_files(backbone)):
                kwargs["use_safetensors"] = False
        except Exception:
            pass  # offline/local path: let from_pretrained decide
        base = AutoModelForCausalLM.from_pretrained(backbone, **kwargs)
        base.requires_grad_(False)  # backbone is always frozen; only LoRA trains

        # GPT-2-style backbones (the smoke-test tiny-gpt2) have no q_proj/v_proj
        module_names = {n.split(".")[-1] for n, _ in base.named_modules()}
        targets = [t for t in lora_targets if t in module_names] or ["c_attn"]
        self.lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                                   lora_dropout=lora_dropout,
                                   target_modules=targets, bias="none",
                                   task_type="CAUSAL_LM")
        self.model = get_peft_model(base, self.lora_cfg)
        if not load_4bit:
            self.model = self.model.to(device)
        # keep LoRA weights in fp32 even under a half-precision backbone: fp16
        # adapter training diverges without loss scaling, and peft casts the
        # activations to the adapter dtype inside LoraLayer.forward anyway
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
        self.device = device
        self.llm_dim = self.model.config.hidden_size

        # "Yes"/"No" scoring ids. BPE tokenizers (GPT-2) need the leading-space
        # variant after "#Answer:"; SentencePiece (Llama/Vicuna) strips it anyway.
        self.yes_id = self._first_id(" Yes", "Yes")
        self.no_id = self._first_id(" No", "No")

    def _first_id(self, *variants: str) -> int:
        for v in variants:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return self.tokenizer.encode(variants[-1], add_special_tokens=False)[0]

    # ------------------------------------------------------------------ #
    def _embed_text(self, text: str, add_bos: bool) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos and self.tokenizer.bos_token_id is not None:
            ids = [self.tokenizer.bos_token_id] + ids
        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self.model.get_input_embeddings()(ids)  # [T, llm_dim]

    def _pack(self, rows: list[torch.Tensor]):
        """LEFT-pad variable-length embedding rows into one batch.

        Left padding puts every row's answer position at index -1, so the
        P("Yes") readout is a single slice, and (with position_ids computed from
        the attention mask) is numerically identical to unpadded inference.
        """
        maxlen = max(r.size(0) for r in rows)
        dt = rows[0].dtype
        embeds = torch.zeros(len(rows), maxlen, self.llm_dim, dtype=dt, device=self.device)
        attn = torch.zeros(len(rows), maxlen, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            embeds[i, maxlen - r.size(0):] = r
            attn[i, maxlen - r.size(0):] = 1
        pos = (attn.cumsum(dim=1) - 1).clamp(min=0)
        return embeds, attn, pos

    def forward(self, his_titles: list[list[str]], target_titles: list[str],
                user_tokens: torch.Tensor | None = None,
                item_tokens: torch.Tensor | None = None,
                warm_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Returns P("Yes") in [0, 1], shape [B].

        Hybrid mode when user_tokens/item_tokens are given (Phase 2 / eval);
        text-only mode when both are None (Phase 1 warm-up).
        warm_tokens (SeLLa-Rec arm): [B, 1, llm_dim] <WarmID> tokens appended
        after the item token via the SEG_W clause.
        """
        hybrid = user_tokens is not None
        dt = self.model.get_input_embeddings().weight.dtype
        rows = []
        for b in range(len(target_titles)):
            if hybrid:
                a, m, c = fill_titles(his_titles[b], target_titles[b], hybrid=True)
                parts = [
                    self._embed_text(a, add_bos=True),
                    user_tokens[b].to(dt),          # N x <UserID>
                    self._embed_text(m, add_bos=False),
                    item_tokens[b].to(dt),          # 1 x <TargetItemID>
                ]
                if warm_tokens is not None:         # SeLLa arm: 1 x <WarmID>
                    parts += [self._embed_text(SEG_W, add_bos=False),
                              warm_tokens[b].to(dt)]
                parts.append(self._embed_text(c, add_bos=False))
                row = torch.cat(parts, dim=0)
            else:
                text = fill_titles(his_titles[b], target_titles[b], hybrid=False)
                row = self._embed_text(text, add_bos=True)
            rows.append(row)

        embeds, attn, pos = self._pack(rows)
        out = self.model(inputs_embeds=embeds, attention_mask=attn, position_ids=pos)
        logits = out.logits[:, -1, :]                       # next-token logits at answer slot
        yes_no = torch.stack([logits[:, self.yes_id], logits[:, self.no_id]], dim=-1)
        return torch.softmax(yes_no.float(), dim=-1)[:, 0]  # P("Yes")

    # ------------------------------------------------------------------ #
    def lora_state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.model.state_dict().items() if "lora_" in k}

    def load_lora_state_dict(self, sd: dict):
        missing = self.model.load_state_dict(sd, strict=False)
        unexpected = [k for k in missing.unexpected_keys]
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"

    def trainable_lora_parameters(self):
        return [p for n, p in self.model.named_parameters() if "lora_" in n]

    def set_lora_trainable(self, flag: bool):
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(flag)
