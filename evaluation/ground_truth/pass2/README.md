# Ground truth, pass 2: evidence (15 October 2026)

These scripts and outputs back the pass-2 entry in `docs/mapping_protocol.md` §6. They
were written for pass 2 and do not use any pass-1 notes. Run them from the repository
root with the sources below. Paths are command-line arguments. They are kept as run,
so ruff skips this folder.

## Sources (fresh downloads, 15 October 2026)

| Source | Where | SHA-256 |
|---|---|---|
| HAI technical details PDF | `icsdataset/hai` @ `2a814ce`, `hai_dataset_technical_details.pdf` | `0668345c…a556`, identical to the copy pass 1 used |
| HAI 21.03 test1–5 | same commit, `hai-21.03/test*.csv.gz` | the decompressed CSVs match `data/raw_checksums.json`, the laptop's data |
| CIC Modbus 2023 attack pcaps | the project's raw data | match `data/raw_checksums.json` |
| ATT&CK for ICS | `mitre-attack/attack-stix-data` @ `c2494b2`, `ics-attack/ics-attack-20.0.json` | pinned in `src/agmesh/evaluation/attack_ics.py` |

## Steps and outputs

1. **`parse_pdf.py`** transcribes the "HAI 21.03, Attack operation" table from
   `pdftotext -layout` output into `pdf_rows.json`.
   - The table is on printed pp. 36–38 (PDF pages 37–39).
   - The pairing of combination rows and the merged date cells were checked against
     the rendered pages.
2. **`diff_code.py`** compares `pdf_rows.json` field by field with `TIMETABLE`. Output:
   `diff_code.txt`, 0 differences in 50 rows.
3. **`segments.py`** finds the labelled attack segments in the raw test CSVs. Output:
   `labelled_segments.json`, 50 segments.
4. **`diff_data.py`** matches each segment to the documented attack nearest in start
   time, without using the project's `match()`. Outputs: `diff_data.txt`,
   `data_check.json`.
   - 50 of 50 match.
   - Start offsets are −1 to +21 s.
   - The labelled processes agree for all 50.
   - Durations agree within 5 s for 49. The exception is A209: 199 s labelled, 159 s
     documented.
5. **`points.py`** checks that the documented target points exist and that they move
   during the attacks. Output: `target_points.txt`.
   - Three document names are not 21.03 columns: `P2_AutoSD`, `P2_SCO` and
     `P2_LCV01D`.
   - Controller outputs and frozen sensors move in most events.
   - No setpoint column moves in any setpoint event. The document says setpoint
     changes are hidden in the HMI.
6. **`modbus_episodes.py`** dissects Modbus/TCP independently, keeping only the
   attacker's (185.175.0.7) requests in each Site B episode window. Output:
   `site_b_episodes.txt`.
