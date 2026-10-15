import json,sys,time,threading,urllib.request
N=int(sys.argv[1]); U="http://localhost:8000/v1/chat/completions"; bar=threading.Barrier(N); res=[None]*N
def one(i):
    b={"model":"nemotron-3.5-super-vl","messages":[{"role":"user","content":f"Write a long story about lighthouse #{i}."}],"max_tokens":512,"ignore_eos":True,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request(U,json.dumps(b).encode(),{"Content-Type":"application/json"}); bar.wait(); t0=time.time(); first=None; n=0
    with urllib.request.urlopen(r,timeout=900) as f:
        for line in f:
            line=line.decode().strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"): continue
            d=json.loads(line[6:])
            if d.get("usage"): n=d["usage"]["completion_tokens"]
            if first is None and any((c.get("delta") or {}).get("content") for c in d.get("choices",[])): first=time.time()
    res[i]=(t0,first,time.time(),n)
ts=[threading.Thread(target=one,args=(i,)) for i in range(N)]; [t.start() for t in ts]; [t.join() for t in ts]
t0=min(r[0] for r in res); t1=max(r[2] for r in res); tot=sum(r[3] for r in res)
per=[r[3]/(r[2]-r[1]) for r in res]; tt=[r[1]-r[0] for r in res]
print(f"S{N} aggregate={tot/(t1-t0):.1f} tok/s per-stream={sum(per)/N:.1f} tok/s ttft={min(tt):.2f}-{max(tt):.2f}s tokens={tot}")
