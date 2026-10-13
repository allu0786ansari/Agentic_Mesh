# Phase 0 (Week 1) — Laptop Runbook

This runbook covers everything for Week 1 that must happen on your own 8 GB laptop.
The code is already on the `phase3` branch. Your job is to set up the environment,
run the three feasibility measurements, and commit their results. Those numbers feed
Gate G0 (Implementation Plan v3.0 §7), which fixes the decisions every later phase
depends on.

**Time needed:** about 2 hours, mostly waiting for downloads and benchmarks.
**You will commit:** `uv.lock`, `data/raw_checksums.json`, `data/raw.dvc`, `data/.gitignore` and four files in `results/feasibility/`.

> Run every command in this runbook inside **WSL2 Ubuntu** unless a step says PowerShell.
> Do not run the LLM benchmark and the Ray benchmark at the same time: each one measures memory, and running them together spoils both measurements.

---

## Step 1 — Configure WSL2 memory (PowerShell, once)

1. If Ubuntu 24.04 is not installed yet, run this in PowerShell as Administrator, restart if asked, and create your Linux user:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

2. Open the WSL configuration file:

   ```powershell
   notepad $env:USERPROFILE\.wslconfig
   ```

3. Paste this, then save. If your laptop has fewer than 4 CPU cores, set `processors` to your core count.

   ```ini
   [wsl2]
   memory=6GB
   processors=4
   swap=4GB
   ```

4. Restart WSL:

   ```powershell
   wsl --shutdown
   ```

5. Open Ubuntu again and check that the memory total is about 5.8 Gi:

   ```bash
   free -h
   ```

## Step 2 — Install the tools (Ubuntu, once)

```bash
sudo apt update
sudo apt install -y git make curl rsync build-essential gh

# uv (Python and dependency manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12

# DVC (data versioning), installed as an isolated tool
uv tool install dvc

# Ollama (local LLM server)
curl -fsSL https://ollama.com/install.sh | sh
```

Then sign in to GitHub so you can push from WSL:

```bash
gh auth login        # choose GitHub.com, HTTPS, and sign in with the browser
gh auth setup-git
git config --global user.name  "Allaudin Ansari"
git config --global user.email "<the email on your GitHub account>"
```

## Step 3 — Get the code

Keep the repository on the Linux filesystem (under `~`), not under `/mnt/c`. Files under `/mnt/c` are much slower to read from WSL.

```bash
cd ~
git clone https://github.com/allu0786ansari/Agentic_Mesh.git agentic-mesh
cd agentic-mesh
git checkout phase3
```

## Step 4 — Install dependencies and check the code

```bash
make setup        # first run resolves dependencies and creates uv.lock
make check        # ruff, mypy and pytest
```

**Expected result:** Ruff reports `All checks passed!`, mypy reports `Success: no issues found`, and pytest ends with every test passing and **1 skipped**. The skipped test needs the federated-learning stack, which you install next.

Now install the federated-learning stack. This downloads Flower, Ray and the CPU build of PyTorch (about 1 GB):

```bash
make setup-fl
uv run --group fl pytest tests/unit/test_ray_bench.py
```

**Expected result:** `4 passed`.

## Step 5 — Copy the raw datasets into WSL

Your HAI and CIC Modbus files are in the old Windows copy of the repository. Copy them in, replacing the path with yours:

```bash
mkdir -p data/raw
rsync -a --info=progress2 "/mnt/c/Users/<you>/<path-to-old-repo>/data/raw/" data/raw/
```

Check the layout:

```bash
find data/raw -maxdepth 3 -type d
```

**Expected result:** you should see these folders:

- `data/raw/hai/hai-21.03`
- `data/raw/modbus2023/benign/{ied1a,ied1b,scada-hmi}`
- `data/raw/modbus2023/attack/{ied1a,ied1b,scada-hmi}`

If your folders are named differently, change `paths.hai_dir` and `paths.modbus_dir` in `configs/params.yaml` to match.

## Step 6 — Record checksums and track the data with DVC

```bash
make checksums      # hashes about 3.7 GB, takes a few minutes
dvc add data/raw    # creates data/raw.dvc
```

**Expected result:** the checksum step ends with `Recorded N files (… GiB)`. If it prints a **WARNING about Git LFS pointers**, those files are placeholders rather than data. Note which ones; the inspection in Step 7 reports them too.

Optional but recommended: give DVC a backup location for the data. It needs about 4 GB free on that drive. For example, to use drive D:

```bash
dvc remote add -d localstore /mnt/d/agmesh-dvc-store
dvc push
```

## Step 7 — Inspect the datasets

```bash
make inspect-data
```

This reads every HAI file and walks every Modbus capture, so it takes a few minutes. When it finishes, it prints a list of **G0 hints**. The most important ones are:

| Hint | What it tells us |
|---|---|
| `hai_usable` | Whether all HAI files were readable (no LFS placeholders) |
| `site_a_partition_scheme` | Whether HAI has distinct operating regimes to split the four Site A nodes by |
| `rq2_fusion_data` | Whether the Modbus attack captures overlap in time across devices (needed for evidence fusion) |
| `site_b_attack_labels` | Whether any attack log or label files exist |
| `site_b_attribution_ground_truth` | Whether the dataset files mention MITRE ATT&CK technique IDs |

## Step 8 — Benchmark the local LLM

Close browsers and other heavy applications first, so the memory numbers reflect the project rather than everything else on the laptop.

```bash
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
ollama pull llama3.2:3b
make bench-llm
```

This sends 31 prompts to each model (one warm-up plus 30 measured) and can take 10–40 minutes. It prints one line per call, then a summary per model with p50 and p95 latency, tokens per second, schema-valid rate, grounded rate and a suggested latency target (L_LLM).

When it finishes, unload the models so the next step measures Ray alone:

```bash
ollama stop phi3.5:3.8b-mini-instruct-q4_K_M
ollama stop llama3.2:3b
ollama ps            # should list no models
```

## Step 9 — Benchmark Flower and Ray

Run the default configuration (4 clients, 3 rounds, 2 CPUs per client), then the 1-CPU variant for comparison:

```bash
make bench-ray
uv run --group fl python scripts/bench_ray.py --num-cpus-per-client 1 \
    --output results/feasibility/ray_bench_1cpu.json
```

Each run takes a few minutes. Each prints whether all rounds completed, the time per round, how many clients ran at once, and peak memory (PSS is the most accurate figure).

## Step 10 — Commit and push the results

```bash
ls results/feasibility/
# data_inspection.json  llm_bench.json  ray_bench.json  ray_bench_1cpu.json

make check
git add uv.lock data/raw_checksums.json data/raw.dvc data/.gitignore results/feasibility/
git commit -m "Week 1: lock file, data checksums and feasibility results"
git push origin phase3
```

If you configured a DVC remote in Step 6, `git add .dvc/config` as well.

Then tell me it's pushed. I'll read the results, fill in `docs/adr/0001-g0-decisions.md`, update `configs/params.yaml` to match, and we'll review Gate G0 together before starting Phase 1.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Run `source "$HOME/.local/bin/env"`, or open a new terminal |
| `make setup` cannot reach `download.pytorch.org` | Check your internet connection or VPN, then re-run; the PyTorch CPU index is needed only to create `uv.lock` |
| `cannot reach Ollama at http://127.0.0.1:11434` | Start it: `sudo systemctl start ollama`, or run `ollama serve` in a second terminal |
| `SKIP <model>: not installed` | Run the `ollama pull` command shown, then re-run `make bench-llm` |
| `free -h` does not show about 6 GB | Check that `.wslconfig` is in your Windows user folder (not in Ubuntu), then `wsl --shutdown` again |
| A benchmark is killed or the laptop freezes | Close other applications. For Ray, re-run with `--clients 2`. Record what happened; it is useful G0 evidence |
| Ray prints warnings about `/dev/shm` or the object store | Usually harmless; the benchmark asks Ray for only 256 MB of object store |
| `make check` fails after you edited a file | Run `make format`, then `make check` again |
| Inspection reports `missing` for HAI or Modbus | The folder paths in `configs/params.yaml` don't match your copy; fix them and re-run |
