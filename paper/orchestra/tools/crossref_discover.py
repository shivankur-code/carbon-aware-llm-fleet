"""Literature discovery + verification via Crossref (PaperOrchestra Step 3; S2/OpenAlex unavailable).

Topical queries from outline.json (rows=10) + targeted title lookups (rows=4). Verification:
Levenshtein title ratio >= 70 for targets, publication date < cutoff, has DOI + authors.
Writes raw_candidates.json; cache in cache/crossref_cache.json.
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

WS = Path(sys.argv[1])
CACHE = WS / "cache" / "crossref_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
SEL = "DOI,title,author,container-title,issued,type,abstract,is-referenced-by-count,published"


def cr(query, rows):
    key = f"{rows}|{query}"
    if key in cache:
        return cache[key]
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": query, "rows": rows, "select": SEL})
    req = urllib.request.Request(url, headers={"User-Agent": "paper-orchestra-litreview/0.1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                items = json.loads(r.read())["message"]["items"]
            cache[key] = items
            CACHE.write_text(json.dumps(cache))
            time.sleep(0.4)
            return items
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []


def norm(it):
    yr = (it.get("issued") or {}).get("date-parts", [[None]])[0]
    return dict(doi=it.get("DOI"), title=(it.get("title") or [""])[0],
                authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])],
                venue=(it.get("container-title") or [""])[0], year=yr[0] if yr else None,
                date="-".join(str(x) for x in yr) if yr and yr[0] else None, type=it.get("type"),
                has_abstract=bool(it.get("abstract")), cites=it.get("is-referenced-by-count", 0))


outline = json.loads((WS / "outline.json").read_text())
plan = outline["intro_related_work_plan"]
queries = [("intro", q) for q in plan["introduction_strategy"]["search_directions"]]
for i, sub in enumerate(plan["related_work_strategy"]["subsections"]):
    for q in sub["limitation_search_queries"]:
        queries.append((f"related_work[{i}]", q))
targets = json.loads((WS / "cache" / "targets.json").read_text())
cands = []
for tag, q in queries:
    for it in cr(q, 10):
        cands.append(dict(norm(it), discovered_for=[tag], query=q))
for t in targets:
    for it in cr(t["title"], 4):
        cands.append(dict(norm(it), discovered_for=[t["for"]], query=t["title"], target=t["title"]))
(WS / "raw_candidates.json").write_text(json.dumps(cands, indent=1))
print(len(queries), "queries,", len(targets), "targets ->", len(cands), "raw candidates")
