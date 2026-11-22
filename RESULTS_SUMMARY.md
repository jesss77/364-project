# Phase 2 Real-Data Results Summary (`phase2-results`)

Here is a summary of everything I completed on this branch: finishing the Clemson preprocessing pipeline on the real archive, running real-data training and hyperparameter tuning across **both datasets (18 total runs)** on Colab GPU, running `int8` quantization and Clemson meal failure analysis, and building a visualization notebook with poster-ready charts so everyone can inspect what happened.

---

## 1. What I Did

1. **Completed Phase 1 for Clemson All-day on real data:**
   - Ran `preprocess_clemson.py`, `validate_clemson_processed.py` (**0 errors**, `VALIDATION PASSED`), and `compute_clemson_normalization.py` on the full `SHMfiles.zip` (`354` recordings, `93,546` 6-minute windows).
   - Added `config/clemson_normalization.json` and the two processed audit files (`clemson_preprocessing_summary.json`, `clemson_processed_validation.json`).

2. **Trained, Tuned & Evaluated KU Leuven FD-I (`13` runs — Q1, Q2, Q4, Q6):**
   - Started with default baselines for all 4 architectures (`cnn`, `cnn_causal`, `cnn_gru`, `transformer`).
   - Diagnosed from validation precision/recall curves that default `--class-weight sqrt` (`[1.0, 10.6, 16.9]`) was causing excessive false-positive bites—especially on `cnn_causal` (`bite P = 0.318`, `bite MAE = 153.0`) and `transformer` (`bite P = 0.468`, `bite MAE = 89.1`).
   - Tuned learning rate, dropout, weight decay, channels, epochs, and class weighting (`--class-weight none`).
   - Evaluated the **4 winning validation configurations** once on the held-out **test split** (strictly keeping test untouched during tuning) and ran static/dynamic **`int8` quantization** (`quantize_fd1.py --eval-test`) on all 4 winners.

3. **Trained, Tuned & Analyzed Clemson All-day (`5` runs — Q3, Q7, Q9):**
   - Trained 5 configurations testing `pos_weight` (`sqrt`, `2.0`, `none`), learning rate, dropout, and weight decay.
   - Selected the best balanced model on validation (`clemson_cnn_pw_none`), scored it on the held-out **test split**, and ran `analyze_clemson_meals.py` on both validation and test for the **Q7** failure breakdown.

4. **Created Visual Analysis & Poster Figures:**
   - Built and ran `results_analysis.ipynb` to visualize all 18 runs, export **6 high-resolution (300 DPI) figures** to `results/figures/`, and generate `results/clemson_model_comparison.csv`.

---

## 2. KU Leuven FD-I Results (Q1, Q2, Q4, Q6)

### Biggest Finding: Class Weighting Ablation (`sqrt` vs. `none`)
Switching from `--class-weight sqrt` to **`--class-weight none`** boosted **every single architecture** on validation because probabilities remained calibrated under `argmax`:

| Architecture | Baseline (`sqrt`) Val Bite F1 / MAE | Tuned (`none`) Val Bite F1 / MAE | Gain |
|---|---|---|---|
| **CNN-GRU** | `0.768` / `22.3` | **`0.822` / `21.3`** | `+0.054` F1 |
| **Dilated CNN** | `0.689` / `27.9` | **`0.820` / `27.4`** | `+0.131` F1 |
| **Causal CNN (0 s delay)** | `0.433` / `153.0` | **`0.734` / `38.7`** | **`+0.301` F1, 4× lower MAE** |
| **Transformer** | `0.549` / `89.1` | **`0.710` / `39.1`** | `+0.161` F1, 2.3× lower MAE |

### Final Unseen Test & `int8` Watch Results (4 Winning Runs)

| Model (`--class-weight none`) | Size (`f32 → int8`) | CPU Speed (`f32`) | Look-ahead | **Test Bite F1** (`f32` / `int8`) | **Test Drink F1** | **Test Bite MAE** (`f32` / `int8`) |
|---|---:|---:|---:|---:|---:|---:|
| **CNN-GRU** (`fd1_cnn_gru_cw_none`) | `0.56 → 0.31 MB` | `103 ms/min` | `15.0 s` | **`0.817`** / **`0.815`** | `0.834` | **`22.0`** / **`22.3`** |
| **Dilated CNN** (`fd1_cnn_cw_none`) | `0.87 → 0.27 MB` | **`5.7 ms/min`** | `7.9 s` | `0.776` / **`0.780`** | **`0.837`** | `24.7` / `23.7` |
| **Causal CNN** (`fd1_cnn_causal_cw_none`) | `0.87 → 0.27 MB` | **`5.7 ms/min`** | **`0.0 s`** | `0.751` / **`0.759`** | `0.787` | `28.7` / `26.7` |
| **Transformer** (`fd1_transformer_cw_none`) | `0.65 → 0.50 MB` | `9.0 ms/min` | `15.0 s` | `0.623` / `0.623` | `0.654` | `46.3` / `46.3` |

**Key Takeaways:**
- **Eating vs. Drinking (Q1):** On the unseen test set, `fd1_cnn_gru_cw_none` had **0 eating-predicted-as-drinking** and **0 drinking-predicted-as-eating**.
- **Eating Speed (Q2):** `fd1_cnn_gru_cw_none` reached **0.62 bites/min MAE** (`1.44 s` between-bite MAE) on validation and **0.86 bites/min MAE** (`1.80 s` between-bite MAE) on unseen test participants.
- **Accuracy vs. Efficiency (Q4 & Q6):** CNN-GRU wins accuracy (`0.817`), Dilated CNN is **18× faster on CPU** (`5.7 ms/min`), and Causal CNN hits `0.751` with **0.0 s delay** for the live demo. Static `int8` shrinks CNN & Causal CNN **3.2× down to 273 KB** with zero accuracy loss.

---

## 3. Clemson All-day Results (Q3, Q7, Q9)

### Validation Comparison (53 Full Days / 710 Hours)

| Run | Best Ep | Val Ep F1 | Meals Found (TPR) | False Alarms / Day | Time F1 | Start / End Err | Meals/Day MAE | Eating Min/Day MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clemson_cnn_base` (`pw=sqrt`) | 6 | `0.748` | `77.6%` | `1.76` | `0.539` | `4.0 / 6.9 m` | `1.00` | `31.8 min` |
| `clemson_cnn_reg` (`pw=sqrt`, reg) | 5 | `0.780` | `77.6%` | `1.25` | `0.502` | `3.7 / 11.1 m` | `0.94` | `42.6 min` |
| `clemson_cnn_pw2_lr5e4` (`pw=2`) | 6 | `0.744` | `74.1%` | `1.49` | `0.575` | `3.3 / 5.4 m` | `1.08` | `25.1 min` |
| **`clemson_cnn_pw_none` (Selected)** | 10 | **`0.761`** | `74.1%` | **`1.22`** | **`0.586`** | **`3.1 / 5.6 m`** | **`0.92`** | **`22.2 min`** |
| `clemson_cnn_pw_none_reg` | 3 | `0.697` | `62.1%` | `0.95` | `0.591` | `2.4 / 4.3 m` | `1.28` | `19.5 min` |

*Why I chose `clemson_cnn_pw_none`:* Although `clemson_cnn_reg` had slightly higher val Episode F1 (`0.780`), it stretched meal end boundaries (`11.1 min` end error, Time F1 `0.502`, predicting `78.8` eating min/day vs. true `42.1`). `clemson_cnn_pw_none` gave the strongest overall balance across Episode F1, Time F1, false alarms/day, and daily totals.

### Unseen Test Results (`clemson_cnn_pw_none` — 53 Full Days / 706 Hours / 165 Meals)
- **Episode F1:** `0.694` (**`71.5%`** of meals detected: `118/165`; **`1.94` false alarms/day** = `0.08/hr`)
- **Time F1:** `0.551` | **Boundary Errors:** Start `4.1 min`, End `6.5 min`
- **Daily Totals (Q9):** True `3.11` vs. Predicted `3.30` meals/day (**MAE `1.02`**); True `47.9` vs. Predicted `57.6` eating min/day (**MAE `25.9 min`**)

### Failure Analysis Highlights (Q7 — `meal_breakdown_test.json`)
- **Short meals / snacks are missed most:** Detection drops smoothly with duration — `30+ min` (**`85.7%`**), `15–30 min` (**`79.2%`**), `5–15 min` (**`71.1%`**), **`< 5 min` (`45.5%`)**. Similarly, Lunch (`79.2%`) and Dinner (`75.0%`) outperform Snacks (`63.0%`), since 6-minute windows dilute brief eating episodes.
- **Left-handed drop:** Right-handed recordings (`Right`/`R`) achieved **`74.3%`** detection (`110/148`), whereas Left-handed (`Left`) dropped to **`33.3%`** (`4/12`) — likely due to right-hand dominance in training and mirrored left-wrist IMU axes.
- **Age & Context:** Ages `25–40` hit **`87.8%`** (`1.35` FP/day) vs. `under 25` at **`62.2%`** (`2.37` FP/day); `Restaurant` & `Office` both reached **`80.0%`**.

---

## 4. Files Added on This Branch

- `config/clemson_normalization.json` — Clemson training-only per-channel mean/std
- `results/data_audit/clemson_preprocessing_summary.json` & `clemson_processed_validation.json` — Clemson processed data audit (0 errors)
- `results/fd1_model_comparison.csv` & `results/clemson_model_comparison.csv` — Master comparison tables for all 18 runs
- `results/runs/<18 run folders>/` — All `config.json`, `history.json`, `metrics.json`, `predictions_*.csv`, `meals_*.csv`, `quantization.json`, and `meal_breakdown_*.json` files *(trained `*.pt` checkpoints are git-ignored by policy and backed up on Google Drive for sharing)*
- `results_analysis.ipynb` — Rendered Matplotlib analysis notebook viewable directly on GitHub
- `results/figures/01_...png` through `06_...png` — 6 high-resolution (300 DPI) poster-ready charts

---

## 5. What's Next

1. **Q5 (New-User Personalization):** Build `config/fd1_folds.json` from `FD1_7_fold_id_list.zip` and run `personalize_fd1.py` (1 / 2 / 5-minute adaptation).
2. **Live Demo & Wearable Calibration:** Test `run_live.py` + `ui.html` locally using `fd1_cnn_causal_cw_none` (and `clemson_cnn_pw_none`) with replay files and a live phone / Samsung Galaxy Watch via Sensor Logger.
3. **A1 Poster & Pitch Prep:** Use the exported figures in `results/figures/` and prepare the 3–5 minute presentation + Q&A.
