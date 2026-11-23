# Paper via Google's official paper-orchestra

This folder writes the paper with **[google-research/paper-orchestra](https://github.com/google-research/paper-orchestra)**, run **unmodified** and pinned to commit `d9bae99`.

## How it runs without Gemini keys (two backends)
The official pipeline calls Gemini through the `google.genai` SDK. `openrouter_shim.py` replaces that SDK client at runtime with one that sends the same requests to **OpenRouter**, which serves the same Gemini models:
- `gemini-3.1-pro-preview` for writing and reflection;
- `gemini-3-flash-preview` for search;
- `gemini-3-pro-image-preview` for images.

The shim makes these translations:

| Gemini SDK feature | OpenRouter equivalent |
|---|---|
| `system_instruction`, `temperature`, `max_output_tokens` | system message, `temperature`, `max_tokens` |
| `tools=[google_search]` (literature search grounding) | web plugin with `engine: native` (Google Search for Gemini) |
| image output (`response_modalities=[IMAGE]`, aspect ratio) | `modalities: [image, text]`, `image_config.aspect_ratio` |

### Bridge backend (what this run actually used)
OpenRouter returned HTTP 402 (no credit), so the published run used `PO_BACKEND=bridge`:
- The shim writes each model call as `request.md` (the full system and user prompt), `request.json` and any attached images into `$PO_BRIDGE_DIR/<id>/`.
- It then blocks until `response.txt` appears.
- Every request was answered by an **LLM agent**, not Gemini. Writers and reviewers were separate agents, each told to use only the experimental log and the citation map.
- A **replay cache** reuses the answer to a byte-identical prompt, which made restarts cheap. One side effect: the AgentReview "ensemble" of 3 reviews received one review three times, because its three prompts are identical. Each review panel is therefore effectively one reviewer.

### Literature lookups
- Semantic Scholar throttled about 5 of 6 unauthenticated requests (HTTP 429). The shim spaces calls about 1.1 s apart.
- After a 429 it falls back to **Crossref** bibliographic search and returns the result in Semantic Scholar's format. Fallbacks are logged to `output/s2_crossref_fallback.jsonl`.
- Lookups are cached in `output/literature_lookup_cache.json`, so a restart sees the same verified paper set.

Without this, the official code treats a 429 as "not found" and silently drops the citation.

### Other notes
It also spaces Semantic Scholar requests about 1.1 s apart and retries on HTTP 429. The official code otherwise treats a 429 as "not found" and silently drops the citation. Every model call is logged to `output/llm_calls.jsonl` (model, tokens, cost).

## Inputs (`raw_materials/`, official format)
- `idea.md`: the method, with equations (dense variant).
- `experimental_log.md`: every number, copied from `results/`, including the refurbishment study.
- `figures/` + `info.json`: the repo's real result figures. The run uses `--use_plotting false`, so figures are not re-drawn from the text.

## Run
```bash
bash paper/google_po/run.sh                                   # OpenRouter (needs credit)
PO_BACKEND=bridge PO_BRIDGE_DIR=/path/to/bridge bash paper/google_po/run.sh   # answer requests externally
```

It uses the official ICLR 2025 template and a literature cutoff of 2026-10. **Estimated cost:** about USD 10–20, based on roughly 60–70 calls at Gemini 3.1 Pro prices (USD 2 / 12 per million input / output tokens) plus web searches.

Requirements: Python 3.10+, `pdflatex` and `bibtex`, and texlive-science (`algorithm.sty`).

## Result of the published run
- **Paper:** [`output/final_paper.pdf`](output/final_paper.pdf), 22 pages, main text 10 pages, ICLR 2025 template. The LaTeX source is in `output/content_refinement_workdir/final_refined_paper.tex`.
- **Pipeline steps:**
  1. outline;
  2. literature agent: 41 search calls, 21 verified papers;
  3. section writing;
  4. AgentReview, then 3 rounds of refinement by "rebuttal via revision";
  5. a format check and a format fix.
- **Review scores (v0 → v3):** overall 3, 3, 3, 4 (1–10 scale), decision Reject throughout; the reviews are in `output/content_refinement_workdir/peer_reviews/`. The reviewers' main objection was the same each time: GPU throughput, power and the CXL slowdown are placeholders. The GPU measurements in `gpu/` have not been run.
- **Phase 6 came from the reviews.** The v2 and v3 reviews spotted that the reuse host combines a part that saves carbon (reused SSDs) with one that costs it (reused DRAM over CXL). That led to Phase 6 (`paper/notes/P6_FINDINGS.md`): reusing SSDs alone doubles the fleet saving. Phase 6 came after this paper run, so the paper only states it as a hypothesis.
- **Run records:**
  - `output/bridge_archive/bridge_requests_responses.tar.gz`: every prompt and answer (page renders left out);
  - `output/run*.log`: pipeline logs.

The run was restarted three times: twice before the restart below, then once after a container reboot. Replay rules:
- **Identical prompt:** answers were reused automatically by the replay cache.
- **Only the list order changed:** answers were reused by hand (Python set ordering differed between restarts). Each such reuse is logged in `manual_replay.log` inside the archive.

**Known issues in the PDF:**
- Figure 7's axis was labelled MtCO2e; the values are ktCO2e. The figure has been regenerated in `raw_materials/figures/`, but the published PDF still has the old image plus a caption note.
- One Unicode ∞ in a bib title causes a non-fatal LaTeX error.
- The ICLR anonymous mode prints "Anonymous authors".
