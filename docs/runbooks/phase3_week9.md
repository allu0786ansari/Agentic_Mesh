# Phase 3, Week 9 — Laptop Runbook (knowledge base, rendering, retrieval: PT6)

Week 9 is built and tested in the cloud workspace. The protocol is in
`docs/adr/0010-knowledge-base-retrieval-week9.md`, written before any retrieval
result existed. The cloud cannot reach Hugging Face, NIST or the PyTorch index, so
three things happen on your laptop:

1. Updating the lock file for the new `kb` group.
2. Building the knowledge base with the real MiniLM encoder.
3. **The PT6 measurement.**

**Run the retrieval evaluation once.** Its first run is the preregistered result
(`results/retrieval/val__pt6.json`). Any later run is saved as exploratory.

**Time needed:** about 15 minutes, plus downloads of about 100 MB (the model, NIST
and the ATT&CK bundle).

## Step 1 — Update, lock and check

CI's `uv sync --locked` fails until this step's `uv.lock` is pushed. That is
expected, as in Week 6.

```bash
cd ~/Agentic_Mesh
git pull origin main
uv lock                # adds the kb group: qdrant-client, sentence-transformers, pypdf
make setup-kb
make check             # expect 255 passed; the reference-encoder test downloads MiniLM
```

## Step 2 — Sources and the knowledge base (about 5 minutes)

```bash
make kb-sources        # NIST SP 800-82r3 -> data/kb/; SHA-256 recorded in configs/kb/sources.lock.json
make kb                # -> data/kb/qdrant, configs/kb/encoder.lock.json, results/kb/manifest.json
```

`make kb` prints a line like this:

```
104 techniques, 404 passages, NIST <n> passages, dim 384 -> results/kb/manifest.json
```

It must say 104 techniques, 404 passages and dim 384. NIST may be a few hundred
passages.

If NIST still refuses `make kb-sources`, download the PDF by hand, then run
`make kb`:

```bash
curl -L -A "Mozilla/5.0" -o data/kb/NIST.SP.800-82r3.pdf https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf
```

`make kb` records the PDF's SHA-256 either way. The build also works with no NIST
file at all; PT6 does not depend on it.

## Step 3 — Render the queries and compare with the cloud (about 1 minute)

```bash
make retrieval-queries
python3 -c "import json; a=[q['text'] for q in json.load(open('results/retrieval/queries_val.json'))['queries'] if q['kind']=='rendered']; b=[q['text'] for q in json.load(open('results/W9/queries_val.json'))['queries']]; print(len(a), a==b)"
```

The first command should report `18 rendered + 30 paraphrase queries, 1 missed`.
The second compares your 18 rendered texts with the cloud's.

This must print `18 True`. If it prints `False`, stop and tell me before Step 4.

## Step 4 — The PT6 measurement (one run, about 1 minute)

```bash
make retrieval-eval
```

It prints `Recall@5 … MRR … (48 queries; PT6 measurement: True) -> results/retrieval/val__pt6.json`.
Do not run it again.

## Step 5 — Commit and push

```bash
git add uv.lock configs/kb/encoder.lock.json configs/kb/sources.lock.json
git add -f results/kb results/retrieval
git commit -m "Week 9: kb lock, knowledge base build, rendered queries, PT6 retrieval measurement"
git push origin main
```
