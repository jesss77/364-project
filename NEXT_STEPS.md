# Next Steps Toward Final Submission (Due Dec 15, 10:00 AM)

We have completed real-data preprocessing, training, tuning, unseen-test evaluation, `int8` quantization, and failure analysis for **7 of our 8 research questions** (`Q1, Q2, Q3, Q4, Q6, Q7, Q9`). See `RESULTS_SUMMARY.md` and `results_analysis.ipynb` for full metrics and figures.

Only **4 focused tasks** remain:

---

## 1. Test & Calibrate the Live Demo (25% of Poster/Demo Grade)

**Setup choice:** Use a **phone strapped/held on the wrist** running the free **Sensor Logger** app (iOS/Android), pushing live IMU data over a **phone mobile hotspot** to our laptop running `scripts/live/run_live.py` + the browser dashboard (`http://localhost:8050/`).  
*(Note: Galaxy Fit3 runs closed FreeRTOS without raw IMU streaming, so phone + Sensor Logger → laptop is the cleanest, most reliable setup. No mobile app needed — `fd1_cnn_causal_cw_none` runs on laptop CPU at `5.7 ms/min` with `0.0 s` delay.)*

- [ ] Install local dependencies: `pip install -r requirements.txt` and place the downloaded `best.pt` weights into `results/runs/` (from Google Drive).
- [ ] Run the live listener with our winning causal model:
  ```powershell
  python scripts\live\run_live.py --gesture-run fd1_cnn_causal_cw_none --listen 8000 --ui 8050 --show-stream 2 --save my_eating_test.jsonl
  ```
- [ ] In Sensor Logger: enable **Accelerometer**, **Gravity**, **Gyroscope**, **Standardise Units & Frames**, and **HTTP Push** to the printed URL.
- [ ] Verify flat-table gravity reads `~(0, 0, +9.8) m/s²`, then record **2 minutes of real bites & sips**.
- [ ] Replay `my_eating_test.jsonl` to lock in the best `--axes` orientation mapping (e.g., `-y,x,z`) and confirm clean live bite/sip detection.
- [ ] Also test a full-day replay with `--meal-run clemson_cnn_pw_none` so we can demo both models.
- [ ] Record a short backup video (for emergency backup only — live demo is primary).

---

## 2. Finish **Q5** — New-User Personalization (20% "Goals Achieved")

`Q5` (*1 / 2 / 5-minute adaptation on unseen users*) is our **only remaining research question**, and `scripts/train/personalize_fd1.py` is already written.

- [ ] Generate `config/fd1_folds.json` from the official `FD1_7_fold_id_list.zip`.
- [ ] Run on Colab GPU:
  ```bash
  python scripts/train/personalize_fd1.py --model cnn --class-weight none --run-name fd1_personalize_cnn
  ```
  *(Optional: also run with `--model cnn_gru --class-weight none` if GPU time allows.)*
- [ ] Save `personalization.json` to the repo and add one before-vs.-after chart to `results_analysis.ipynb`.

---

## 3. Design & Print the A1 Portrait Poster (25% of Poster/Demo Grade)

- **Format:** PDF, **A1 portrait (`59 × 84 cm`)**, printed and ready to mount **15 min before 10:00 AM on Dec 15**.
- **Minimum font sizes:** Title `≥ 60 pt`, Section headings `≥ 32 pt`, Body `≥ 20 pt`, Figure labels/tables `≥ 18 pt`. Keep text minimal (~2-minute read).
- **Required content:**
  - [ ] Problem & motivation + clean diagram of our two-dataset pipeline & live demo
  - [ ] Dataset summary (KU Leuven FD-I `251.7 h` & Clemson All-day `4,704 h`)
  - [ ] Key results vs. baselines (use the 300-DPI charts in `results/figures/`: `--class-weight none` gain, Q4/Q6 accuracy-speed-`int8` trade-off, Clemson Q3/Q9, and Q5 personalization)
  - [ ] **Successes & Failures:** Successes (`0` eat↔drink confusion, `0.817` Test Bite F1, `273 KB` `int8` model) vs. Failures (Clemson `<5 min` snacks at `45.5%`, left-handed drop to `33.3%`)
  - [ ] Main conclusions, team member names, and course name (`CMPS 364`)

---

## 4. Prepare Pitch & Individual Q&A (30% — Graded Individually!)

- [ ] Rehearse a **3–5 minute pitch** walking through the poster + live demo.
- [ ] Make sure **every team member** can clearly explain any part of the project:
  - Why two independent datasets instead of one
  - Participant-independent splits & train-only normalization (no leakage)
  - Why `--class-weight none` outperformed `sqrt` across all 4 models
  - Trade-offs between CNN-GRU, Dilated CNN, Causal CNN, and Transformer + `int8` compression
  - Hysteresis meal reconstruction & Q7 failure modes (short snacks & left-handed bias)
  - Acknowledgment of any generative AI tools used (per course policy)
