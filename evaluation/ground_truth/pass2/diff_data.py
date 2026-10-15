import json, sys
from datetime import datetime
rows=json.load(open(sys.argv[1]))["rows"]; segs=json.load(open(sys.argv[2]))["segments"]
DATES=[("A101","A105","2020-07-07"),("A201","A206","2020-07-09"),("A207","A220","2020-07-10"),("A301","A306","2020-07-13"),("A307","A308","2020-07-14"),("A401","A405","2020-07-28"),("A501","A507","2020-07-30"),("A508","A512","2020-07-31")]
def date(i): return next(d for a,b,d in DATES if a<=i<=b)
doc=[]
for r in rows:
    st=datetime.strptime(f"{date(r['id'])} {r['time']}","%Y-%m-%d %H:%M")
    procs=sorted({p["controller"].split("-")[0] for p in r["primitives"]})
    doc.append((r["id"],st,r["duration"],procs))
used=set(); res=[]
for s in segs:
    ss=datetime.fromisoformat(s["start"])
    best=min(doc,key=lambda d:abs((d[1]-ss).total_seconds()))
    off=(ss-best[1]).total_seconds()
    res.append((best[0],off,s["seconds"],best[2],s["processes"],best[3],s["file"]))
    used.add(best[0])
print("documented attacks matched:",len(used),"of",len(doc), "| segments:",len(segs))
import statistics
offs=[r[1] for r in res]; print("start offset (data - doc), s: min",min(offs),"max",max(offs),"median",statistics.median(offs))
bad=[r for r in res if abs(r[1])>90 or r[4]!=r[5] or abs(r[2]-r[3])>5]
print("flagged (offset>90 s, process mismatch, or |duration diff|>5 s):",len(bad))
for r in bad: print("  %s offset %+.0fs dur data %d doc %d procs data %s doc %s (%s)"%r)
dd=[r[2]-r[3] for r in res]; print("duration diff data-doc: min",min(dd),"max",max(dd), "count |d|<=5:",sum(abs(x)<=5 for x in dd))
json.dump([dict(zip(["id","offset_s","dur_data","dur_doc","proc_data","proc_doc","file"],r)) for r in res],open(sys.argv[3],"w"),indent=1)
