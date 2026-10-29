"""The three base models: loading, input formatting and the settings every step shares.

  p-IgGen      paired VH+VL, input "1" + sequence + "2", 350 tokens, d_model 768   (Hugging Face ollieturnbull/p-IgGen)
  IgLM         heavy chain, input "[HEAVY][HUMAN]" + sequence, 200 tokens, d_model 512   (installed iglm package)
  ProGen2-OAS  heavy chain, input "1" + sequence + "2", 200 tokens, d_model 1536   (salesforce/progen clone)

NORM_FACTOR: the hyperparameter-matched SAEs read the final-layer-norm output divided by this factor, the root mean
square of the activation norms of their training activations (computed at the start of training; these are the values of the
paper's SAEs)."""
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BertTokenizerFast, GPT2LMHeadModel,
                          PreTrainedTokenizerFast)

from plm_sae import paths as P

PIGGEN = "ollieturnbull/p-IgGen"
PIGGEN_REVISION = "0e5c4ed4e4c6bed0a0c7e56ad6fa24c63ed56023"
MODELS = ("piggen", "IgLM", "progen2_oas")
MAX_LEN = {"piggen": 350, "IgLM": 200, "progen2_oas": 200}
D_MODEL = {"piggen": 768, "IgLM": 512, "progen2_oas": 1536}
NORM_FACTOR = {"piggen": 27.16730499267578, "IgLM": 24.0302, "progen2_oas": 71.1121}


def load(model, device="cpu"):
    """(language model in eval mode on device, tokenizer) for one of MODELS."""
    if model == "piggen":
        lm = AutoModelForCausalLM.from_pretrained(PIGGEN, revision=PIGGEN_REVISION)
        tok = AutoTokenizer.from_pretrained(PIGGEN, revision=PIGGEN_REVISION)
    elif model == "IgLM":
        from iglm.model.IgLM import CHECKPOINT_DICT, VOCAB_FILE
        lm = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DICT["IgLM"])
        tok = BertTokenizerFast(vocab_file=VOCAB_FILE, do_lower_case=False)
        if tok.pad_token is None:
            tok.pad_token = "[PAD]"
            tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")
    elif model == "progen2_oas":
        P.add_progen2_to_syspath()
        from models.progen.modeling_progen import ProGenForCausalLM
        lm = ProGenForCausalLM.from_pretrained(str(P.progen2_dir() / "checkpoints" / "progen2-oas"))
        tok = PreTrainedTokenizerFast(tokenizer_file=str(P.progen2_dir() / "tokenizer.json"))
        tok.pad_token = "<|pad|>"
        tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
    else:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    return lm.to(device).eval(), tok


def _iglm_ids(tok, text, max_len):
    """IgLM tokens: each leading [TAG] is one token, then one token per residue; padded to max_len."""
    tokens, i = [], 0
    while i < len(text) and text[i] == "[":
        end = text.find("]", i)
        if end == -1:
            break
        tokens.append(text[i:end + 1])
        i = end + 1
    ids = tok.convert_tokens_to_ids(tokens + list(text[i:]))[:max_len]
    mask = [1] * len(ids) + [0] * (max_len - len(ids))
    return ids + [tok.pad_token_id] * (max_len - len(ids)), mask


def tokenize(model, tok, seqs, wrap=True, truncation=True, max_len=None):
    """Token tensors {input_ids, attention_mask} for a list of sequences, padded to max_len (default MAX_LEN[model]).
    wrap=True adds the model's start/end or tag tokens; wrap=False takes the texts as they are (the SAE training
    sets of p-IgGen and IgLM already contain them)."""
    n = max_len or MAX_LEN[model]
    if model == "IgLM":
        texts = ["[HEAVY][HUMAN]" + s for s in seqs] if wrap else list(seqs)
        ids, mask = zip(*[_iglm_ids(tok, t, n) for t in texts])
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
    texts = ["1" + s + "2" for s in seqs] if wrap else list(seqs)
    extra = {"add_special_tokens": False} if model == "progen2_oas" else {}
    return tok(texts, padding="max_length", truncation=truncation, max_length=n, return_tensors="pt", **extra)
