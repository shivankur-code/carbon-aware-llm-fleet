# Paper via Google's official paper-orchestra

This folder writes the paper with **[google-research/paper-orchestra](https://github.com/google-research/paper-orchestra)**, run **unmodified** and pinned to commit `d9bae99`.

## How it runs without Gemini keys
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

It also spaces Semantic Scholar requests about 1.1 s apart and retries on HTTP 429. The official code otherwise treats a 429 as "not found" and silently drops the citation. Every model call is logged to `output/llm_calls.jsonl` (model, tokens, cost).

## Inputs (`raw_materials/`, official format)
- `idea.md`: the method, with equations (dense variant).
- `experimental_log.md`: every number, copied from `results/`, including the refurbishment study.
- `figures/` + `info.json`: the repo's real result figures. The run uses `--use_plotting false`, so figures are not re-drawn from the text.

## Run
```bash
bash paper/google_po/run.sh        # needs OpenRouter credit; output -> paper/google_po/output/
```

It uses the official ICLR 2025 template and a literature cutoff of 2026-10. **Estimated cost:** about USD 10–20, based on roughly 60–70 calls at Gemini 3.1 Pro prices (USD 2 / 12 per million input / output tokens) plus web searches.
