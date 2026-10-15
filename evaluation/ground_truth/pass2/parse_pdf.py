"""Pass 2: independent transcription of the HAI 21.03 attack table from pdftotext output."""
import json, re, sys
text = open(sys.argv[1]).read()
sec = text[text.index("HAI 21.03\nHAI 21.03 includes"):]
sec = sec[sec.index("ATTACK OPERATION"):]
ap_re = re.compile(r"(AP\d\d)\s+(P\d-[A-Z]{2}-[A-Z]{2}\d(?:PV\d)?(?:-ST\d*)?)\s+(P\d_\w+(?:, P\d_\w+)?)")
id_re = re.compile(r"^\s*(\d{1,2})\s+(A\d{3})\b")
tail_re = re.compile(r"(\d\d:\d\d)\s+(\d+)\s*$")
items = []  # sequence of ('ap', prim) / ('id', no, id, time, dur, own_ap)
for line in sec.splitlines():
    m_id, m_ap, m_t = id_re.match(line), ap_re.search(line), tail_re.search(line)
    prim = None
    if m_ap:
        prim = {"scenario": m_ap.group(1), "controller": m_ap.group(2), "points": [p.strip() for p in m_ap.group(3).split(",")]}
    if m_id:
        items.append({"kind": "id", "no": int(m_id.group(1)), "id": m_id.group(2), "time": m_t.group(1), "dur": int(m_t.group(2)), "own": prim})
    elif prim:
        items.append({"kind": "ap", "prim": prim, "used": False})
rows = []
for i, it in enumerate(items):
    if it["kind"] != "id": continue
    prims = [it["own"]] if it["own"] else []
    combo = int(it["id"][1]) >= 3
    need = 2 if combo else 1
    if not prims and combo:  # take the unconsumed AP line just before
        j = i - 1
        if j >= 0 and items[j]["kind"] == "ap" and not items[j]["used"]:
            prims.append(items[j]["prim"]); items[j]["used"] = True
    j = i + 1
    while len(prims) < need and j < len(items):
        if items[j]["kind"] == "ap" and not items[j]["used"]:
            prims.append(items[j]["prim"]); items[j]["used"] = True
        elif items[j]["kind"] == "id": break
        j += 1
    rows.append({"no": it["no"], "id": it["id"], "time": it["time"], "duration": it["dur"], "primitives": prims})
leftover = [it for it in items if it["kind"] == "ap" and not it["used"]]
print(json.dumps({"rows": rows, "unassigned_ap_lines": leftover}, indent=1))
