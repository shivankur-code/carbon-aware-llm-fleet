# Research Brief (accumulated cross-agent context)

## §1 · Core Claim and Narrative
_Written by: outline-agent, Step 1_

**Core claim:** For LLM-inference fleets, component reuse survives age-dependent reliability only when lifecycle planning prices reliability in, and it remains beneficial — slightly more so — under guaranteed SLOs, while its magnitude is set by grid carbon intensity rather than by reliability.
**Narrative tension:** Reuse-based server designs are evaluated at a point in time; operators decide sequentially over years with aging reused parts, and GPU hosts are dominated by operational carbon.
**Key novelty framing:** First multi-period, reliability-aware lifecycle optimization of reuse-based servers for AI inference, with break-even conditions (reused-part failure multiplier, cascade energy efficiency) and chance-constrained SLO headroom validated by Monte Carlo.
**Outline decisions:**
- Plotting plan: 6 figures (1 diagram to generate, 5 reused from experiments)
- Related Work clusters: low-carbon server design & embodied accounting; field reliability; carbon-aware scheduling; heterogeneous LLM serving & tiered memory
- Section structure: Introduction, Background and Related Work, Lifecycle Model, Evaluation Methodology, Results, Discussion and Limitations, Conclusion
**Potential weaknesses flagged at outline stage:**
- GPU throughput, power and CXL slowdown are placeholders; cascade and mixed-grid sign conclusions depend on them.
- Most embodied-carbon and failure parameters are assumptions (swept, but not measured).
- Absolute savings on the three-grid fleet are below 1%; the story rests on structure (break-evens, planning effect), which must be framed carefully.

## §2 · Literature Landscape
_Written by: literature-review-agent, Step 3_

**Verified pool:** 53 papers (Crossref DOI metadata 42, arxiv.org 9, USENIX via web search 2); Semantic Scholar was rate-limited and OpenAlex budget-exhausted from this network. Cutoff 2026-10-21; two 2026 papers (KV-cache CXL tiering; KV-cache bit-flip vulnerability) are framed as concurrent work.
**Positioning:** no verified work jointly optimizes multi-year buy/reuse/cascade/retire decisions with age-dependent reliability of reused parts for AI-inference fleets; carbon-aware scheduling is complementary (moves work, not hardware).
**Citation coverage:** 53/53 cited in Introduction + Related Work.
**Caution:** abstracts were mostly unavailable; descriptions of cited works are kept at title/venue level, with no numbers quoted from them.

## §3 · Draft Status
_Written by: section-writing-agent, Step 4_

**Draft:** drafts/paper.tex, 4,644 words, compiles to 18 pages (main text ends on p.12, within the 12-page limit; references excluded).
**Gates:** orphan-cite OK (53/53); anti-leakage OK; latex_sanity reported a brace mismatch that is a false positive (escaped \{ \} in the reserve-update formula; braces balance and the document compiles without errors); claim-evidence: 74/89 supported, 15 flagged are rounding/unit artefacts (e.g. 5.77 vs 5.766, 2.7% vs 0.0272, en-dash ranges parsed as negatives), each checked by hand; figure lint: 3 low-resolution warnings (275-300 dpi).
**Known weaknesses for refinement:** placeholder GPU parameters; small absolute fleet savings; no measured C2/C3 results.
