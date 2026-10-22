# Phase 5, Week 1 — Download and verify the new data (ADR 0021)

Read `docs/adr/0021-improvement-phase-protocol.md` §1 first.

**The one rule: never open the final data.** That means `data/raw/hai/hai-23.05/`
and `data/raw/modbus2023/compromised-scada/`.

- **Not allowed:** `head`, `less`, a spreadsheet, a notebook, or a Wireshark look.
- **Allowed:** downloading, copying and hashing.

The project's code refuses to read those paths before `g5-frozen`.

**Disk space:** about 1.5 GB for HAI, plus the CIC captures (several GB). Check
`df -h ~` first.

## Step 1 — Update

```bash
cd ~/Agentic_Mesh
git pull origin main
uv sync --locked --group dev --group ml --group fl --group kb --group figures
```

## Step 2 — HAI 22.04 and 23.05 (from GitHub, via Git LFS)

```bash
sudo apt-get install -y git-lfs && git lfs install
cd ~ && GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/icsdataset/hai.git hai_src
cd ~/hai_src && git lfs pull --include "hai-22.04/*,hai-23.05/*"
mkdir -p ~/Agentic_Mesh/data/raw/hai/hai-22.04 ~/Agentic_Mesh/data/raw/hai/hai-23.05
cp hai-22.04/*.csv ~/Agentic_Mesh/data/raw/hai/hai-22.04/
cp hai-23.05/* ~/Agentic_Mesh/data/raw/hai/hai-23.05/
cd ~/Agentic_Mesh && make phase5-verify
```

**Expected:** `20/20 HAI files verified`. It hashes each file against the checksum
GitHub published; it does not read them. Then delete the clone: `rm -rf ~/hai_src`.

## Step 3 — CIC Modbus 2023: the compromised scenarios

Open http://cicresearch.ca/CICDataset/CICModbusDataset2023/ in a browser and find
the attack folders for the **compromised IED** and the **compromised SCADA HMI**.
Download each folder whole, captures and attack logs, keeping its inner structure:

| Download | Put it in |
|---|---|
| compromised IED | `~/Agentic_Mesh/data/raw/modbus2023/compromised-ied/` |
| compromised SCADA HMI | `~/Agentic_Mesh/data/raw/modbus2023/compromised-scada/` (**do not open**) |

If the site offers the folders as an archive, that is fine. Extract it straight
into the target folder without browsing it.

```bash
cd ~/Agentic_Mesh && make phase5-record
```

It records each file's SHA-256 in `data/raw_checksums_phase5.json`.

## Step 4 — Commit and push

```bash
git add -f data/raw_checksums_phase5.json
git commit -m "Phase 5: new data downloaded and verified (ADR 0021)"
git push origin main
```

Tell me the folder and file names that `make phase5-record` printed for the
compromised-IED scenario. The Site B pipeline will read that one.
