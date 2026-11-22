"""Literature discovery + verification against Semantic Scholar (PaperOrchestra Step 3, Phases 1-2).

Runs (a) topical discovery queries from outline.json and (b) targeted title lookups,
throttled to ~1 QPS with an on-disk cache. Writes raw_candidates.json with S2 metadata.
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

WS = Path(sys.argv[1])
CACHE = WS / "cache" / "s2_cache.json"
FIELDS = "paperId,title,year,venue,publicationDate,authors,abstract,externalIds,citationCount,publicationVenue"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def s2(query, limit):
    key = f"{limit}|{query}"
    if key in cache:
        return cache[key]
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?" +
           urllib.parse.urlencode(dict(query=query, limit=limit, fields=FIELDS)))
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                data = json.loads(r.read())
            cache[key] = data.get("data", [])
            CACHE.write_text(json.dumps(cache))
            time.sleep(1.1)
            return cache[key]
        except Exception as e:  # 429 / transient
            time.sleep(3 * (attempt + 1))
    return []


outline = json.loads((WS / "outline.json").read_text())
plan = outline["intro_related_work_plan"]
queries = [("intro", q) for q in plan["introduction_strategy"]["search_directions"]]
for i, sub in enumerate(plan["related_work_strategy"]["subsections"]):
    for q in sub["limitation_search_queries"]:
        queries.append((f"related_work[{i}]", q))
targets = json.loads((WS / "cache" / "targets.json").read_text())   # [{"title":..., "for":...}]

cands = []
for tag, q in queries:
    for p in s2(q, 8):
        cands.append(dict(p, discovered_for=[tag], query=q))
for t in targets:
    for p in s2(t["title"], 3):
        cands.append(dict(p, discovered_for=[t["for"]], query=t["title"], target=t["title"]))
(WS / "raw_candidates.json").write_text(json.dumps(cands, indent=1))
print(len(queries), "queries,", len(targets), "targets ->", len(cands), "raw candidates")
