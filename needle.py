import json,sys,time,random,urllib.request
W=int(sys.argv[1]); random.seed(W); w="alpha beta gamma delta river stone cloud paper metal glass".split()
body=[random.choice(w) for _ in range(W)]; body.insert(int(W*0.6)," The secret code is PURPLE-4419. ")
p=" ".join(body)+"\nWhat is the secret code? Answer with the code only."
b={"model":"nemotron-3.5-super-vl","messages":[{"role":"user","content":p}],"max_tokens":30,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}
r=urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(b).encode(),{"Content-Type":"application/json"})
t0=time.time();first=None;txt="";u=None
with urllib.request.urlopen(r,timeout=7200) as f:
    for line in f:
        line=line.decode().strip()
        if not line.startswith("data: ") or line.endswith("[DONE]"): continue
        d=json.loads(line[6:]); u=d.get("usage") or u
        for c in d.get("choices",[]):
            s=(c.get("delta") or {}).get("content") or ""
            if s and first is None: first=time.time()
            txt+=s
p=u["prompt_tokens"]; tt=(first or time.time())-t0
print(f"NEEDLE ctx={p} ttft={tt:.1f}s prefill={p/tt:.0f} t/s answer={txt.strip()!r} {'CORRECT' if 'PURPLE-4419' in txt else 'INCORRECT'}")
