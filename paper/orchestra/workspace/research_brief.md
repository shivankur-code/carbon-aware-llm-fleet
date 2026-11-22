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
