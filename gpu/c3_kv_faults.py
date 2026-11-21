"""C3b - Can the KV cache live on aged, reused DRAM? Bit-flip fault injection into the KV cache.

The KV cache of an LLM server is *recomputable state*. If it sits in reused (older, flakier)
DRAM, what do memory errors cost?

For each sample: prefill a context, corrupt the cached K/V tensors with independent bit flips
at a given bit-error rate (BER), then teacher-force a continuation and measure
  * perplexity increase vs the clean cache,
  * top-1 next-token agreement with the clean run,
  * fraction of KV blocks that must be recomputed (detected errors).

Protection schemes, applied per flip pattern:
  none      every flip lands silently
  secded    64-bit-word SECDED ECC (standard server DRAM): 1 flip/word corrected,
            2 flips/word detected -> that KV block recomputed, >=3 flips/word silent
  checksum  per-block checksum (block = 16 tokens x one layer's K or V): any flip -> recompute

Optional ``--bit-class`` restricts flips to sign / exponent / mantissa bits (which bits matter?).

Runs on GPU or CPU (use a small model on CPU for testing), with Hugging Face transformers.
    python gpu/c3_kv_faults.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --out gpu/results
    python gpu/c3_kv_faults.py --model HuggingFaceTB/SmolLM2-135M --device cpu --samples 2 --context 256   # smoke test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

BLOCK_TOKENS = 16
BIT_CLASSES = {  # bit positions (LSB = 0) within one element
    torch.bfloat16: dict(sign=[15], exponent=list(range(7, 15)), mantissa=list(range(0, 7))),
    torch.float16: dict(sign=[15], exponent=list(range(10, 15)), mantissa=list(range(0, 10))),
    torch.float32: dict(sign=[31], exponent=list(range(23, 31)), mantissa=list(range(0, 23))),
}
INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


# ---------------------------------------------------------------- cache access (transformers 4.x and 5.x)
def kv_tensors(cache) -> list[torch.Tensor]:
    """Return the list of K and V tensors (shape [batch, heads, seq, head_dim]) held by a cache."""
    if hasattr(cache, "layers"):  # transformers >= 4.56 / 5.x
        out = []
        for layer in cache.layers:
            out += [layer.keys, layer.values]
        return out
    if hasattr(cache, "key_cache"):
        return [t for pair in zip(cache.key_cache, cache.value_cache) for t in pair]
    return [t for pair in cache for t in pair]  # legacy tuple-of-tuples


# ---------------------------------------------------------------- fault injection
def inject(tensors: list[torch.Tensor], ber: float, protection: str, rng: np.random.Generator,
           bit_class: str = "all") -> dict:
    """Flip bits in-place. Returns counts of flips, silent flips and recomputed blocks."""
    stats = dict(flips=0, silent=0, corrected=0, recomputed_blocks=0, total_blocks=0)
    if not tensors:
        return stats
    dt = tensors[0].dtype
    width = torch.finfo(dt).bits
    allowed = list(range(width)) if bit_class == "all" else BIT_CLASSES[dt][bit_class]
    elems_per_word = 64 // width
    for t in tensors:
        b, h, s, d = t.shape
        n_el = t.numel()
        n_blocks = b * h * math.ceil(s / BLOCK_TOKENS)
        stats["total_blocks"] += n_blocks
        n_bits = n_el * len(allowed)
        k = rng.binomial(n_bits, ber) if ber > 0 else 0
        if k == 0:
            continue
        stats["flips"] += k
        el = rng.integers(0, n_el, size=k)
        bit = np.array(allowed)[rng.integers(0, len(allowed), size=k)]
        if protection == "secded":
            word = el // elems_per_word
            uniq, inv, counts = np.unique(word, return_inverse=True, return_counts=True)
            per_flip = counts[inv]
            keep = per_flip >= 3                              # silent (beyond SECDED)
            stats["corrected"] += int((per_flip == 1).sum())
            detected = el[per_flip == 2]
            stats["recomputed_blocks"] += len(set(_block_id(detected, t.shape)))
            el, bit = el[keep], bit[keep]
        elif protection == "checksum":
            stats["recomputed_blocks"] += len(set(_block_id(el, t.shape)))
            el, bit = el[:0], bit[:0]
        elif protection != "none":
            raise ValueError(protection)
        stats["silent"] += len(el)
        if len(el):
            flat = t.view(-1).view(INT_VIEW[dt])
            idx = torch.as_tensor(el, device=t.device)
            masks = torch.as_tensor((1 << bit.astype(np.int64)), device=t.device)
            if width == 16:
                masks = torch.where(masks >= 2**15, masks - 2**16, masks)
            elif width == 32:
                masks = torch.where(masks >= 2**31, masks - 2**32, masks)
            masks = masks.to(flat.dtype)
            # apply flips one index at a time where duplicated, vectorized otherwise
            uniq, first = np.unique(el, return_index=True)
            if len(uniq) == len(el):
                flat[idx] ^= masks
            else:
                for i, m in zip(idx.tolist(), masks.tolist()):
                    flat[i] ^= m
    return stats


def _block_id(el: np.ndarray, shape) -> np.ndarray:
    b, h, s, d = shape
    tok = (el // d) % s
    bh = el // (s * d)
    return bh * 10_000_000 + tok // BLOCK_TOKENS


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def continuation_nll(model, cache, last_logits, cont_ids):
    out = model(cont_ids, past_key_values=cache, use_cache=True)
    logits = torch.cat([last_logits[:, None, :], out.logits[:, :-1, :]], dim=1).float()
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), cont_ids.reshape(-1), reduction="none")
    return nll, logits.argmax(-1)


def load_text(source: str, text_file: str | None) -> str:
    if text_file:
        return Path(text_file).read_text()
    from datasets import load_dataset
    if source == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(x for x in ds["text"] if x.strip())
    raise ValueError(source)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--continuation", type=int, default=256)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--trials", type=int, default=3, help="independent fault draws per sample and BER")
    ap.add_argument("--bers", default="0,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3")
    ap.add_argument("--protections", default="none,secded,checksum")
    ap.add_argument("--bit-classes", default="all", help="comma list of all,sign,exponent,mantissa")
    ap.add_argument("--text-source", default="wikitext")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="gpu/results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    dtype = getattr(torch, a.dtype)
    text = load_text(a.text_source, a.text_file)
    if a.model == "tiny-random":
        # Offline smoke test: random-init tiny Llama, byte-level tokens. Numbers are meaningless;
        # it only exercises the code path (cache access, injection, protection accounting).
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(a.seed)
        cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
        model = LlamaForCausalLM(cfg).to(dtype).to(dev).eval()
        ids = torch.tensor(list(text.encode("utf-8", "ignore")), dtype=torch.long)
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
        ids = tok(text, return_tensors="pt").input_ids[0]
    L = a.context + a.continuation
    starts = np.linspace(0, len(ids) - L - 1, a.samples).astype(int)
    bers = [float(x) for x in a.bers.split(",")]
    prots = a.protections.split(",")
    classes = a.bit_classes.split(",")
    rng = np.random.default_rng(a.seed)
    rows = []
    t0 = time.time()
    for si, st in enumerate(starts):
        ctx = ids[st: st + a.context][None].to(dev)
        cont = ids[st + a.context: st + L][None].to(dev)
        with torch.no_grad():
            pre = model(ctx, use_cache=True)
        base_cache = pre.past_key_values
        last = pre.logits[:, -1, :]
        clean_nll, clean_top = continuation_nll(model, copy.deepcopy(base_cache), last, cont)
        clean = float(clean_nll.mean())
        for ber in bers:
            for prot in (prots if ber > 0 else ["none"]):
                for bc in (classes if ber > 0 else ["all"]):
                    for trial in range(a.trials if ber > 0 else 1):
                        cache = copy.deepcopy(base_cache)
                        st_ = inject(kv_tensors(cache), ber, prot, rng, bc)
                        nll, top = continuation_nll(model, cache, last, cont)
                        m = float(nll.mean())
                        rows.append(dict(sample=si, ber=ber, protection=prot, bit_class=bc, trial=trial,
                                         nll=m if math.isfinite(m) else float("inf"), clean_nll=clean,
                                         ppl_ratio=math.exp(min(m - clean, 50)) if math.isfinite(m) else float("inf"),
                                         top1_agree=float((top == clean_top).float().mean()),
                                         recompute_frac=st_["recomputed_blocks"] / max(1, st_["total_blocks"]),
                                         **{k: v for k, v in st_.items() if k != "total_blocks"}))
        print(f"sample {si + 1}/{len(starts)} done ({time.time() - t0:.0f}s), clean ppl {math.exp(clean):.2f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby(["protection", "bit_class", "ber"]).agg(
        ppl_ratio_median=("ppl_ratio", "median"), ppl_ratio_p90=("ppl_ratio", lambda x: float(np.quantile(x, 0.9, method="higher"))),
        top1_agree_mean=("top1_agree", "mean"), recompute_frac_mean=("recompute_frac", "mean"),
        silent_flips_mean=("silent", "mean"), n=("ppl_ratio", "size")).reset_index()
    print(g.to_string())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = a.label or (torch.cuda.get_device_name(0).replace(" ", "_") if dev == "cuda" else "cpu")
    stem = Path(a.out) / f"c3faults_{tag}_{a.model.split('/')[-1]}_{a.dtype}"
    df.to_csv(f"{stem}_raw.csv", index=False)
    g.to_csv(f"{stem}_summary.csv", index=False)
    Path(f"{stem}_meta.json").write_text(json.dumps(dict(
        kind="c3_kv_faults", args=vars(a), host=platform.node(), device=dev, torch=torch.__version__,
        block_tokens=BLOCK_TOKENS, seconds=time.time() - t0), indent=2))
    print("wrote", stem)


if __name__ == "__main__":
    main()
