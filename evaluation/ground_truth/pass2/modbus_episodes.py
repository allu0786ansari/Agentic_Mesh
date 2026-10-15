"""Pass 2: independent Modbus/TCP dissection of the attack-day pcaps (pure Python)."""
import struct, sys, json, collections, datetime as dt
EP = [(0,"03:51:37","03:52:46"),(1,"03:58:25","04:01:29"),(2,"04:04:19","04:05:58"),(3,"04:08:17","04:09:17"),
      (4,"04:17:05","04:37:34"),(5,"13:03:42","13:08:01"),(6,"13:21:52","13:22:13")]
day = dt.date(2023,2,1)
def ts(s): return dt.datetime.combine(day, dt.time.fromisoformat(s), tzinfo=dt.timezone.utc).timestamp()
EPS = [(e, ts(a)-1, ts(b)+1) for e,a,b in EP]
FC = {1:"read coils",2:"read discrete inputs",3:"read holding registers",4:"read input registers",5:"write single coil",6:"write single register",15:"write multiple coils",16:"write multiple registers"}
def packets(path):
    f = open(path,"rb"); gh = f.read(24); magic = struct.unpack("<I", gh[:4])[0]
    e = "<" if magic in (0xa1b2c3d4,0xa1b23c4d) else ">"; nano = magic in (0xa1b23c4d,0x4d3cb2a1)
    link = struct.unpack(e+"I", gh[20:24])[0]
    while True:
        h = f.read(16)
        if len(h) < 16: return
        s, us, incl, orig = struct.unpack(e+"IIII", h); data = f.read(incl)
        yield s + us/(1e9 if nano else 1e6), data, link, orig
def modbus(data, link):
    off = 14 if link == 1 else 16 if link == 113 else 0
    if link == 1:
        et = struct.unpack(">H", data[12:14])[0]
        if et == 0x8100: et = struct.unpack(">H", data[16:18])[0]; off = 18
        if et != 0x0800: return None
    ip = data[off:]; 
    if len(ip) < 20 or ip[0]>>4 != 4 or ip[9] != 6: return None
    ihl = (ip[0]&15)*4; src = ".".join(map(str,ip[12:16])); dst = ".".join(map(str,ip[16:20]))
    tcp = ip[ihl:]; sp, dp = struct.unpack(">HH", tcp[:4]); thl = (tcp[12]>>4)*4
    total = struct.unpack(">H", ip[2:4])[0]; pl = tcp[thl:total-ihl]
    if 502 not in (sp, dp) or not pl: return None
    frames = []; i = 0
    while i + 8 <= len(pl):
        tid, pid, ln, uid = struct.unpack(">HHHB", pl[i:i+7]); fc = pl[i+7]
        body = pl[i+8:i+6+ln]
        frames.append({"fc": fc, "len": ln, "body": body, "avail": len(pl)-i-6})
        if ln < 2: break
        i += 6 + ln
    return {"src": src, "dst": dst, "req": dp == 502, "frames": frames, "payload": len(pl)}
out = {}
seen = set()
for path in sys.argv[1:]:
    for t, data, link, orig in packets(path):
        ep = next((e for e,a,b in EPS if a <= t <= b), None)
        if ep is None: continue
        m = modbus(data, link)
        if m is None or not m["req"] or m["src"] != "185.175.0.7": continue
        o = out.setdefault(ep, {"requests":0, "fc":collections.Counter(), "sources":collections.Counter(), "dsts":collections.Counter(),
              "max_quantity":0, "addresses":collections.Counter(), "multi_frame_packets":0, "max_frames":0, "len_mismatch":0, "truncated":0, "first":t, "last":t, "file":collections.Counter()})
        o["first"] = min(o["first"], t); o["last"] = max(o["last"], t); o["file"][path.split("/")[-2]] += 1
        o["sources"][m["src"]] += 1; o["dsts"][m["dst"]] += 1
        o["packets"]=o.get("packets",0)+1
        if len(m["frames"]) > 1: o["multi_frame_packets"] += 1
        o["max_frames"] = max(o["max_frames"], len(m["frames"]))
        for fr in m["frames"]:
            o["requests"] += 1; o["fc"][FC.get(fr["fc"], f"fc{fr['fc']}")] += 1
            if fr["len"] - 0 != len(fr["body"]) + 2: o["len_mismatch"] += 1
            b = fr["body"]
            if len(b) >= 4 and fr["fc"] in (1,2,3,4,15,16):
                addr, q = struct.unpack(">HH", b[:4]); o["max_quantity"] = max(o["max_quantity"], q); o["addresses"][(fr["fc"],addr)] += 1
            elif len(b) >= 4 and fr["fc"] in (5,6):
                addr, v = struct.unpack(">HH", b[:4]); o["addresses"][(fr["fc"],addr)] += 1
            elif len(b) < 4: o["truncated"] += 1
for ep in sorted(out):
    o = out[ep]
    print(f"episode {ep}: {o.get('packets')} packets, {o['requests']} request frames, {dt.datetime.utcfromtimestamp(o['first']).time()}–{dt.datetime.utcfromtimestamp(o['last']).time()}, captured on {dict(o['file'])}")
    print(f"   fc {dict(o['fc'])}; max quantity {o['max_quantity']}; multi-frame packets {o['multi_frame_packets']} (max {o['max_frames']} frames); MBAP-length mismatches {o['len_mismatch']}; truncated {o['truncated']}")
    print(f"   sources {dict(o['sources'])} -> {dict(o['dsts'])}; distinct (fc,address) {len(o['addresses'])}: {dict(list(o['addresses'].most_common(4)))}")
