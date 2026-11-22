# Paper bundle (input for paper-orchestra)

Everything a paper-writing pipeline needs. Generated files are rebuilt from results with:

```bash
python paper/build_bundle.py
```

| File | What it is | Generated? |
|---|---|---|
| `outline.md` | Title, abstract (filled with numbers), section plan, figure list with captions | yes |
| `claims.md` | Every claim → evidence file → figure → status (PROVISIONAL / MEASURED / TO-VERIFY) | yes |
| `numbers.json` | Every quoted number with its source file | yes |
| `brief_headline.md` | The X / Y / Z sentence and Figure 1 for the research brief | yes |
| `parameters.md` | All model parameters with value, range, unit, status, source | yes (`experiments/p1_node_lifecycle.py`) |
| `related_work.md` | Related-work draft with citation keys | hand-written |
| `refs.bib` | BibTeX, 53 entries verified against Crossref DOI metadata, arxiv.org or USENIX pages (copied from the PaperOrchestra run) | generated |
| `notes/P1..P4_FINDINGS.md` | Phase-by-phase findings with reproduction commands | hand-written |
| `figures/` | All figures (PNG, 150 dpi) | yes (experiment scripts) |

## Rules for the writer
1. Quote numbers only from `numbers.json` or `claims.md`. Keep each claim's status visible. Anything PROVISIONAL must be described as model-based with placeholder GPU parameters until `configs/measured.yaml` exists.
2. State the carbon-accounting convention (cut-off, amortized) and show the sensitivity table (claim C9).
3. This is a modelling study using public parameters. Say so, and say that no proprietary fleet data was used.
4. Every `assumption` parameter cited as a fact must be checked first.

## Refresh after the GPU runs
```bash
python gpu/ingest.py --c2-old ... --c2-new ... --c3-local ... --c3-remote ...   # writes configs/measured.yaml
python experiments/p1_node_lifecycle.py && python experiments/p1_tornado.py --region mixed
python experiments/p3_study.py all && python experiments/p4_chance.py
python paper/build_bundle.py
```
