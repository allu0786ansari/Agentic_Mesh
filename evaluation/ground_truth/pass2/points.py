import pandas as pd, numpy as np, json, sys, glob
rows=json.load(open(sys.argv[1]))["rows"]; chk={r["id"]:r for r in json.load(open(sys.argv[2]))}
segs=json.load(open(sys.argv[3]))["segments"]; cols=set(json.load(open(sys.argv[3]))["columns"])
pts=sorted({p for r in rows for pr in r["primitives"] for p in pr["points"]})
print("documented points absent from the data columns:",[p for p in pts if p not in cols])
cand={p:sorted(c for c in cols if "_" in c and (c.split("_",1)[1].lower()[:3]==p.split("_",1)[1].lower()[:3] or c.endswith(p.split("_",1)[1]))) for p in pts if p not in cols}
print("same-suffix candidates:",cand)
dfs={f.split("/")[-1]:pd.read_csv(f,parse_dates=["time"]).set_index("time") for f in sorted(glob.glob(sys.argv[4]+"/test*.csv.gz"))}
ALIAS={"P2_AutoSD":"P2_ASD","P2_SCO":"P2_CO_rpm","P2_LCV01D":"P3_LCV01D"}
def moved(df,col,start,secs):
    s=pd.Timestamp(start); pre=df[col][s-pd.Timedelta("10min"):s-pd.Timedelta("1s")]; dur=df[col][s:s+pd.Timedelta(seconds=secs)]
    med=pre.median(); mad=(pre-med).abs().median(); dev=(dur-med).abs().max()
    scale=max(mad*1.4826, 1e-6*max(1,abs(med)), (pre.max()-pre.min())/2)
    return dev/scale
out=[]
for r in rows:
    c=chk[r["id"]]
    seg=[s for s in segs if s["file"]==c["file"] and s["seconds"]==c["dur_data"]]
    df=dfs[c["file"]]
    # data start = documented + offset
    import datetime
    for pr in r["primitives"]:
        for p in pr["points"]:
            col=ALIAS.get(p,p)
            # find segment start from data_check offset
            out.append((r["id"],pr["controller"],p,col))
# recompute starts from segments directly via offsets
DATES=[("A101","A105","2020-07-07"),("A201","A206","2020-07-09"),("A207","A220","2020-07-10"),("A301","A306","2020-07-13"),("A307","A308","2020-07-14"),("A401","A405","2020-07-28"),("A501","A507","2020-07-30"),("A508","A512","2020-07-31")]
date=lambda i: next(d for a,b,d in DATES if a<=i<=b)
res={}
for aid,ctrl,p,col in out:
    r=next(x for x in rows if x["id"]==aid); c=chk[aid]
    st=pd.Timestamp(f"{date(aid)} {r['time']}")+pd.Timedelta(seconds=c["offset_s"])
    z=moved(dfs[c["file"]],col,st,c["dur_data"])
    kind="SP" if ctrl.split("-")[2].startswith("SP") else "CO"
    role="sensor(PV)" if p in ("P1_PIT01","P1_FT03","P1_LIT01","P2_SIT01") else kind
    res.setdefault((role,col),[]).append(round(float(z),1))
for k,v in sorted(res.items()): print(k, "n=%d"%len(v), "moved(>5)=%d"%sum(x>5 for x in v), v)
