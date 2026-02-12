# Mobile Evaluation

This document collects *all* on-device training/peak-RAM measurements. It lists the original results reported in the paper (Google Pixel 6) and the newly collected measurements (Samsung Galaxy S21 FE 5G and additional Pixel 6 TinyLlama RAM measurements). All measurements are reported as **mean ± standard deviation**.

---

## Measurement methodology

* Discard the first warm-up step. Continue training for **nine forward–backward passes with optimizer updates**. Measure the time to complete a single training step (forward+backward+update). Repeat the entire process **five times** and report the mean and standard deviation of the per-step time.
* Reported values for training **step time** are in **seconds** (mean ± std).
* **Peak RAM usage** is reported in **GB**.
* **Batch size (bs):** All experiments used **bs = 4**, *except* the Qwen measurement on Samsung Galaxy S21 FE 5G which used **bs = 1** due to OOM at larger batch sizes (this affects the per-step timing and should be noted when comparing numbers).

---

## Devices / hardware

* **Google Pixel 6** — 8 GB RAM (~ 6.7 GB available)
* **Samsung Galaxy S21 FE 5G** — 6 GB RAM (~ 2.9 GB available)

---

## Units and naming

* Timing values: **seconds per training step** (mean ± std)
* Peak RAM: **GB**
* Methods: **LoRA**, **MARS OPT0**, **MARS Q-OPT1** (MARS OPT1)
* Ranks reported as `r` (2, 8, 32)

---

## Training step time

### Google Pixel 6

**Models:** SmolLM2 (360M) and Qwen-2 (0.5B).

**SmolLM2 (360M) — step time (s)**

| Rank        |     LoRA (s) | MARS Q-OPT0 (s) | MARS Q-OPT1 (s) |
| ----------- | -----------: | --------------: | --------------: |
| r=2         | 18.94 ± 0.12 |    18.49 ± 0.22 |    18.07 ± 0.12 |
| r=8         | 19.12 ± 0.22 |    19.39 ± 0.17 |    18.33 ± 0.07 |
| r=32        | 21.94 ± 0.11 |    21.29 ± 0.04 |    19.76 ± 0.14 |
| **Average** |    **20.01** |       **19.72** |       **18.75** |

**Qwen-2 (0.5B) — step time (s)**

| Rank        |     LoRA (s) | MARS Q-OPT0 (s) | MARS Q-OPT1 (s) |
| ----------- | -----------: | --------------: | --------------: |
| r=2         | 21.87 ± 0.10 |    21.55 ± 0.12 |    20.43 ± 0.20 |
| r=8         | 22.11 ± 0.19 |    21.64 ± 0.13 |    20.96 ± 0.08 |
| r=32        | 23.74 ± 0.21 |    22.10 ± 0.04 |    21.55 ± 0.23 |
| **Average** |    **22.57** |       **21.76** |       **20.98** |


### Samsung Galaxy S21 FE 5G

**Notes:** SmolLM2 measurements used **bs = 4**. Qwen-2 measurements on this device used **bs = 1** due to OOM at larger batch sizes; timing for Qwen is therefore not directly comparable to bs=4 measurements.

**SmolLM2 (bs = 4) — step time (s)**

| Rank        |     LoRA (s) | MARS OPT0 (s) | MARS OPT1 (s) |
| ----------- | -----------: | ------------: | ------------: |
| r=2         | 16.20 ± 0.10 |  15.05 ± 0.08 |  14.44 ± 0.18 |
| r=8         | 18.72 ± 0.11 |  15.82 ± 0.22 |  15.30 ± 0.33 |
| r=32        | 23.10 ± 0.43 |  17.81 ± 0.51 |  16.10 ± 0.39 |
| **Average** |    **19.34** |     **16.22** |     **15.28** |

**Qwen-2 (bs = 1 on Samsung) — step time (s)**

| Rank        |     LoRA (s) | MARS OPT0 (s) | MARS OPT1 (s) |
| ----------- | -----------: | ------------: | ------------: |
| r=2         |  6.91 ± 0.07 |   6.75 ± 0.02 |   6.39 ± 0.19 |
| r=8         |  7.38 ± 0.09 |   7.17 ± 0.11 |   6.96 ± 0.21 |
| r=32        | 10.10 ± 0.31 |   9.29 ± 0.38 |   8.98 ± 0.43 |
| **Average** |     **8.13** |      **7.74** |      **7.44** |

> **Important:** Qwen timings above were measured with **bs = 1** on Samsung due to OOM at larger batch sizes; these numbers will appear faster compared to bs=4 runs and must be treated accordingly when comparing across devices/experiments.

---

## Peak RAM usage measurements

**All reported peak memory is the observed maximum resident memory during a training step (GB). Ranks reported where available.**

| Device (RAM)                    | Model             | Rank | LoRA Peak RAM | MARS OPT0 Peak RAM | MARS OPT1 Peak RAM |
| ------------------------------- | ----------------- | ---: | ------------: | -----------------: | -----------------: |
| Google Pixel 6 (8 GB)           | TinyLlama         | r=32 |        2.8 GB |             2.8 GB |             2.6 GB |
| Samsung Galaxy S21 FE 5G (6 GB) | SmolLM2 | r=32 |        2.4 GB |             2.4 GB |             2.2 GB |

---

## Important notes / summary of measurement caveats

* **Batch size:** Unless otherwise noted, measurements used **bs = 4**. The Qwen run on Samsung used **bs = 1** (OOM otherwise) — this influences step-time comparisons.
* **Repetitions:** Each reported mean/std is computed from 5 independent repeats of the 9-step timing procedure (first warm-up step discarded), measuring one-step duration each repeat.
* **Devices:** Original paper values were collected on **Google Pixel 6 (8GB)**. New measurements were collected on **Samsung Galaxy S21 FE 5G (6GB)** except the TinyLlama Peak-RAM entry which was measured on Google Pixel 6.
* **Units:** Step time in **seconds** (mean ± std). Peak RAM in **GB**.