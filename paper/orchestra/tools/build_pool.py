"""Build citation_pool.json + refs.bib from the verified set (Crossref DOIs, arXiv ids, USENIX papers)."""
import json, re, sys, time, urllib.request
from pathlib import Path

WS = Path(sys.argv[1])
UA = {"User-Agent": "paper-orchestra-litreview/0.1"}
DOI = [  # key, doi, discovered_for
 ("wang2024greensku", "10.1109/isca59077.2024.00041", "intro"),
 ("gupta2022act", "10.1145/3470496.3527408", "intro"),
 ("gupta2022chasing", "10.1109/mm.2022.3163226", "intro"),
 ("strubell2019energy", "10.18653/v1/p19-1355", "intro"),
 ("chien2023genai", "10.1145/3604930.3605705", "intro"),
 ("tannu2023ssd", "10.1145/3630614.3630616", "related_work[0]"),
 ("switzer2023junkyard", "10.1145/3575693.3575710", "related_work[0]"),
 ("zhang2024spatialembodied", "10.1145/3632775.3661939", "related_work[0]"),
 ("acun2023carbonexplorer", "10.1145/3575693.3575754", "related_work[0]"),
 ("poonia2024circular", "10.1108/meq-06-2023-0185", "related_work[0]"),
 ("schroeder2009dram", "10.1145/1555349.1555372", "related_work[1]"),
 ("sridharan2012dram", "10.1109/sc.2012.13", "related_work[1]"),
 ("sridharan2015memory", "10.1145/2775054.2694348", "related_work[1]"),
 ("meza2015dram", "10.1109/dsn.2015.57", "related_work[1]"),
 ("meza2015flash", "10.1145/2796314.2745848", "related_work[1]"),
 ("zivanovic2019dram", "10.1145/3357526.3357558", "related_work[1]"),
 ("dramicdcs2024", "10.1109/icdcs60910.2024.00152", "related_work[1]"),
 ("ssdsystor2016", "10.1145/2928275.2928278", "related_work[1]"),
 ("ssdnas2024", "10.1109/nas63802.2024.10781343", "related_work[1]"),
 ("hochschild2021cores", "10.1145/3458336.3465297", "related_work[1]"),
 ("sdctaco2024", "10.1145/3690825", "related_work[1]"),
 ("kokolis2025reliability", "10.1109/hpca61900.2025.00096", "related_work[1]"),
 ("wiesner2021letswait", "10.1145/3464298.3493399", "related_work[2]"),
 ("radovanovic2023cics", "10.1109/tpwrs.2022.3173250", "related_work[2]"),
 ("hanafy2024carbonscaler", "10.1145/3673660.3655048", "related_work[2]"),
 ("souza2023ecovisor", "10.1145/3575693.3575709", "related_work[2]"),
 ("sukprasert2024limitations", "10.1145/3627703.3650079", "related_work[2]"),
 ("li2024sprout", "10.18653/v1/2024.emnlp-main.1215", "related_work[2]"),
 ("kwon2023vllm", "10.1145/3600006.3613165", "related_work[3]"),
 ("patel2024splitwise", "10.1109/isca59077.2024.00019", "related_work[3]"),
 ("patel2024power", "10.1145/3620666.3651329", "related_work[3]"),
 ("stojkovic2025dynamollm", "10.1109/hpca61900.2025.00102", "related_work[3]"),
 ("li2023pond", "10.1145/3575693.3578835", "related_work[3]"),
 ("maruf2023tpp", "10.1145/3582016.3582063", "related_work[3]"),
 ("sun2023demystifying", "10.1145/3613424.3614256", "related_work[3]"),
 ("berger2023cxlpooling", "10.1109/mm.2023.3237491", "related_work[3]"),
 ("caseagainstcxl2023", "10.1145/3626111.3628195", "related_work[3]"),
 ("rwcache2026", "10.1145/3833142.3833181", "related_work[3]"),
 ("bitflipkv2026", "10.5220/0015138800004103", "related_work[3]"),
 ("li2017dnnerrors", "10.1145/3126908.3126964", "related_work[3]"),
 ("zadeh1965fuzzy", "10.1016/s0019-9958(65)90241-x", "method"),
 ("huangfu2018highs", "10.1007/s12532-017-0130-5", "method"),
]
ARXIV = [
 ("llama3", "2407.21783", "related_work[1]"),
 ("griggs2024melange", "2404.14527", "related_work[3]"),
 ("wu2022sustainableai", "2111.00364", "intro"),
 ("patterson2021carbon", "2104.10350", "intro"),
 ("luccioni2023bloom", "2211.02001", "intro"),
 ("faiz2024llmcarbon", "2309.14393", "intro"),
 ("zhong2024distserve", "2401.09670", "related_work[3]"),
 ("stojkovic2024greener", "2403.20306", "related_work[3]"),
 ("dixit2021sdc", "2102.11245", "related_work[1]"),
]
USENIX = """@inproceedings{schroeder2016flash,
  title = {Flash Reliability in Production: The Expected and the Unexpected},
  author = {Schroeder, Bianca and Lagisetty, Raghav and Merchant, Arif},
  booktitle = {14th USENIX Conference on File and Storage Technologies (FAST 16)},
  pages = {67--80},
  year = {2016}
}

@inproceedings{pinheiro2007disk,
  title = {Failure Trends in a Large Disk Drive Population},
  author = {Pinheiro, Eduardo and Weber, Wolf-Dietrich and Barroso, Luiz Andr{\\'e}},
  booktitle = {5th USENIX Conference on File and Storage Technologies (FAST 07)},
  year = {2007}
}
"""


def get(url, accept=None):
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    for a in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60).read().decode()
        except Exception:
            time.sleep(2 * (a + 1))
    raise RuntimeError(url)


bib, pool = [], []
for key, doi, tag in DOI:
    b = get(f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex")
    b = re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@%s{%s," % (m.group(1), key), b.strip(), count=1)
    b = b.replace("<sub>", "").replace("</sub>", "")
    bib.append(b)
    meta = json.loads(get(f"https://api.crossref.org/works/{doi}"))["message"]
    pool.append(dict(bibtex_key=key, title=meta["title"][0], year=meta["issued"]["date-parts"][0][0],
                     venue=(meta.get("container-title") or [""])[0],
                     authors=[{"name": f"{a.get('given', '')} {a.get('family', '')}".strip()} for a in meta.get("author", [])],
                     externalIds={"DOI": doi}, verified_via="crossref", match_score=100, discovered_for=[tag]))
    time.sleep(0.4)
for key, aid, tag in ARXIV:
    html = get(f"https://arxiv.org/abs/{aid}")
    meta = lambda n: re.findall(r'<meta name="%s" content="([^"]*)"' % n, html)  # noqa: E731
    title = meta("citation_title")[0].replace("&#39;", "'").replace("M\\'elange", "M{\\'e}lange")
    authors = meta("citation_author")
    year = meta("citation_date")[0][:4]
    auth = " and ".join(authors) if len(authors) <= 30 else " and ".join(authors[:30]) + " and others"
    bib.append("@misc{%s,\n  title = {%s},\n  author = {%s},\n  year = {%s},\n  eprint = {%s},\n  archivePrefix = {arXiv},\n  howpublished = {arXiv:%s}\n}" % (key, title, auth, year, aid, aid))
    pool.append(dict(bibtex_key=key, title=title, year=int(year), venue="arXiv", authors=[{"name": a} for a in authors[:30]],
                     externalIds={"ArXiv": aid}, verified_via="arxiv.org", match_score=100, discovered_for=[tag]))
    time.sleep(1)
bib.append(USENIX.strip())
pool += [dict(bibtex_key="schroeder2016flash", title="Flash Reliability in Production: The Expected and the Unexpected", year=2016,
              venue="FAST", authors=[{"name": n} for n in ("Bianca Schroeder", "Raghav Lagisetty", "Arif Merchant")],
              externalIds={}, verified_via="web_search (usenix.org)", match_score=100, discovered_for=["related_work[1]"]),
         dict(bibtex_key="pinheiro2007disk", title="Failure Trends in a Large Disk Drive Population", year=2007, venue="FAST",
              authors=[{"name": n} for n in ("Eduardo Pinheiro", "Wolf-Dietrich Weber", "Luiz Andre Barroso")],
              externalIds={}, verified_via="web_search (usenix.org)", match_score=100, discovered_for=["related_work[1]"])]
(WS / "refs.bib").write_text("\n\n".join(bib) + "\n")
n = len(pool)
(WS / "citation_pool.json").write_text(json.dumps(dict(papers=pool, cutoff_date="2026-10-21", min_cite_paper_count=int(0.9 * n),
    verification_note="Semantic Scholar was rate-limited and OpenAlex budget-exhausted from this network; papers were verified "
                      "against Crossref DOI metadata, arxiv.org pages, or USENIX pages via web search."), indent=1))
print(n, "papers; min cite", int(0.9 * n))
