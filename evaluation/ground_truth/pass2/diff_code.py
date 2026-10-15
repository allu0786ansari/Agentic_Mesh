import json, sys
sys.path.insert(0, "src")  # run from the repository root
from agmesh.evaluation.hai_attacks import TIMETABLE
rows = json.load(open(sys.argv[1]))["rows"]
# date cells as read from the rendered pages (merged cells)
DATES = [("A101","A105","2020-07-07"),("A201","A206","2020-07-09"),("A207","A220","2020-07-10"),
         ("A301","A306","2020-07-13"),("A307","A308","2020-07-14"),("A401","A405","2020-07-28"),
         ("A501","A507","2020-07-30"),("A508","A512","2020-07-31")]
def date(i): return next(d for a,b,d in DATES if a<=i<=b)
code = {a.attack_id: a for a in TIMETABLE}
diffs = []
for r in rows:
    a = code.get(r["id"])
    if a is None: diffs.append((r["id"], "missing in code")); continue
    start = f"{date(r['id'])} {r['time']}"
    if a.start != start: diffs.append((r["id"], "start", a.start, start))
    if a.duration_s != r["duration"]: diffs.append((r["id"], "duration", a.duration_s, r["duration"]))
    pdf = sorted((p["controller"], tuple(p["points"])) for p in r["primitives"])
    cod = sorted((p.code, tuple(p.points)) for p in a.primitives)
    if pdf != cod: diffs.append((r["id"], "primitives", cod, pdf))
extra = set(code) - {r["id"] for r in rows}
print("code attacks", len(code), "pdf rows", len(rows), "extra in code", sorted(extra))
print("differences:", len(diffs))
for d in diffs: print(" ", d)
