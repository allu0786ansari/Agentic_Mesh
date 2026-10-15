import pandas as pd, json, sys, glob
out=[]; cols=None
for f in sorted(glob.glob(sys.argv[1]+"/test*.csv.gz")):
    df=pd.read_csv(f); cols=list(df.columns)
    t=pd.to_datetime(df["time"]); a=df["attack"].values
    i=0; n=len(a)
    while i<n:
        if a[i]==1:
            j=i
            while j<n and a[j]==1: j+=1
            seg=df.iloc[i:j]
            procs=[p for p in ("P1","P2","P3") if seg[f"attack_{p}"].max()==1]
            out.append({"file":f.split("/")[-1],"start":str(t.iloc[i]),"end":str(t.iloc[j-1]),"rows":j-i,
                        "seconds":(t.iloc[j-1]-t.iloc[i]).total_seconds()+1,"processes":procs})
            i=j
        else: i+=1
json.dump({"segments":out,"columns":cols},open(sys.argv[2],"w"),indent=1)
print(len(out),"segments"); print([c for c in cols if c.startswith(("time","attack"))])
