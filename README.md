# Replicating and Extending IMM-TSF: A Benchmarking Study on Irregular Multimodal EPA-Air Quality Data

**Advisor:** Dr. Bin Luo, Assistant Professor of Data Science

---

## Project Overview

This project replicates and extends the **IMM-TSF** benchmark framework (Chang et al., NeurIPS 2025 Datasets & Benchmarks Track) on the **EPA-Air** dataset — one of nine datasets in the TIME-IMM collection. EPA-Air represents the *Multi-Source Asynchrony* irregularity type, where four environmental sensors (AQI, Ozone, PM2.5, Temperature) report at independent, asynchronous schedules across eight U.S. counties.

The work is organized into two tracks:

- **Track A (Reproducibility):** Replicate the paper's Table 11 baseline results using the official IMM-TSF library. We successfully ran 7 of the 11 models (DLinear, Informer, PatchTST, TimesNet, TimeMixer, TTM, tPatchGNN) and attempted the remaining 4 (TimeLLM, CRU, LatentODE, NeuralFlow), documenting the failure modes encountered.
- **Track C (Methodological Extension):** Three targeted ablations that probe the paper's architectural claims — (1) GPT-2 vs BERT encoder swap, (2) architecture family × encoder interaction analysis, and (3) a placebo test with distribution-matched noise embeddings.

---

## Key Results

### Track A — Reproduction (7 of 11 models)

| Model | Type | Ours Uni | Paper Uni | Ours Multi | Paper Multi | Δ% (Ours) | Text Helps? |
|---|---|---|---|---|---|---|---|
| DLinear | Regular | 0.5090 | 0.5361 | 0.4960 | 0.5223 | −2.6% | ✓ |
| TimesNet | Regular | 0.5680 | 0.5599 | 0.6010 | 0.5892 | +5.8% | ✗ |
| Informer | Regular | 0.6430 | 0.6301 | 0.5900 | 0.5812 | −8.3% | ✓ |
| TimeMixer | Regular | 0.6190 | 0.6086 | 0.5740 | 0.5641 | −7.3% | ✓ |
| PatchTST | Regular | 0.6410 | 0.6196 | 0.6580 | 0.6204 | +2.7% | ✗ |
| TTM | Large TS | 0.5980 | 0.6002 | 0.6340 | 0.6218 | +6.0% | ✗ |
| tPatchGNN* | Irregular | 0.6710 | 0.6258 | 0.6380 | 0.5840 | −5.0% | ✓ |
| **Average** | | **0.6070** | **0.6115** | **0.5987** | **0.5976** | **−2.1%** | |

*\*tPatchGNN run with `patch_size=4` due to incompatibility with default `patch_size=24` on EPA-Air's sparse features.*

Average MSE reduction from text integration: **−2.1%** (vs paper's −6.7% cross-dataset average). The gap is explained by EPA-Air's weak text-to-sensor semantic alignment and the subset of models tested.

### Track A — Remaining 4 Models (attempted)

| Model | Status | Failure Mode |
|---|---|---|
| TimeLLM | Partial success | Default patch_size=24 incompatible with EPA-Air's sparse features; resolved with reduced patch size |
| CRU | Failed | ODE solver instability on EPA-Air's 1.02-hour inter-observation interval over a 7-day horizon |
| LatentODE | Failed | Gradient explosion from accumulated numerical errors in continuous-time integration |
| NeuralFlow | Failed | Same ODE solver instability as CRU/LatentODE |

### Track C — Extension Findings

1. **Encoder choice matters on EPA-Air** — switching from GPT-2 to BERT produced performance swings of up to 13.3 percentage points for individual models (TTM: +6.0% → −7.3%), directly challenging the paper's Figure 6c claim that encoder choice is negligible.
2. **Architecture family predicts text benefit** — non-patch models (DLinear, Informer, TimesNet, TimeMixer) benefit from text with both encoders (avg −3%), while patch-based models (PatchTST, TTM, tPatchGNN) are harmed by GPT-2 (+4.6%) but helped by BERT (−3.0%).
3. **Placebo test** — only 3 of 7 models show genuine semantic benefit from real text. Two models show a placebo effect (noise performs within 1% of real text), and two are actively harmed by real text more than by noise.

---

## Repository Structure

```
├── README.md                              # This file
├── DM-ll_PROJECT_REPORT_.pdf              # Final 19-page project report
│
├── notebooks/
│   ├── DM-ll_PROJECT_CODE.ipynb           # Track A (7 models) + Track C ablations
│   │                                      #   GPT-2 vs BERT, arch family analysis,
│   │                                      #   placebo test, visualizations
│   │
│   ├── Extention.ipynb                    # Per-county evaluation extension
│   │                                      #   Spearman correlation + cluster-bootstrap CI
│   │                                      #   across 8 counties × 7 models
│   │
│   ├── DM_ll_Project_7_Models.ipynb       # Independent 7-model replication + Track C
│   │                                      #   (same experiments, separate execution)
│   │
│   └── DM_ll_Project_4_Remaining_Models.ipynb  # TimeLLM, CRU, LatentODE, NeuralFlow
│                                          #   attempts with failure documentation
│
└── results/                               # (on Google Drive)
    └── runs.jsonl                         # All logged experiment results
```

---

## Reproducing the Experiments

### Prerequisites

- Google Colab Pro with NVIDIA A100 GPU (40GB VRAM)
- Kaggle account (for dataset download)
- Google Drive (for embedding cache persistence across sessions)

### Step-by-step

1. **Open the notebook** in Google Colab (any of the 4 notebooks; `DM-ll_PROJECT_CODE.ipynb` is the most comprehensive).

2. **Run the setup cells** (cells 1–8 in `DM-ll_PROJECT_CODE.ipynb`):
   - Mounts Google Drive
   - Restores Kaggle credentials
   - Clones the official IMM-TSF and Time-IMM repositories
   - Installs PyTorch 2.7.0 + required dependencies (`reformer_pytorch`, `stribor`, `geotorch`)
   - Downloads EPA-Air dataset from Kaggle
   - Restores pre-computed GPT-2 text embeddings from Drive cache

3. **Track A — Run the 7 baseline models:**
   - Cell 19: Regular models (Informer, DLinear, PatchTST, TimesNet, TimeMixer) × uni + multi
   - Cell 20: TTM × uni + multi
   - Cell 22: tPatchGNN × uni + multi (with `patch_size=4`)
   - Cell 23: Results table + comparison with paper Table 11

4. **Track C — Run ablation experiments:**
   - Cells 29–30: Precompute BERT embeddings + cache to Drive
   - Cell 31: Run all 7 models with BERT encoder
   - Cell 32: Encoder comparison table + visualization
   - Cell 33: Architecture family × encoder interaction analysis
   - Cells 34–36: Placebo test (generate noise embeddings, run experiments)
   - Cell 37: Placebo results table + visualization

5. **Remaining 4 models** (use `DM_ll_Project_4_Remaining_Models.ipynb`):
   - Cell 12: TimeLLM (succeeds with modified patch size)
   - Cells 14–20: CRU, LatentODE, NeuralFlow (document failure modes)
   - Cells 21–27: Combined 11-model results table

### Common CLI pattern for all model runs

```bash
python main.py \
  --dataset EPA-Air \
  --data_root /content/IMM-TSF/data \
  --model {MODEL_NAME} \
  --history 7 \
  --pred_window 7 \
  --stride 7 \
  --time_unit days \
  --split_method sample \
  --seed 42 \
  --gpu 0 \
  --batch_size 8 \
  --lr 1e-3 \
  --epoch 50 \
  --patience 10
```

For multimodal runs, add:
```bash
  --enable_text \
  --use_text_embeddings \
  --TTF_module TTF_RecAvg \
  --MMF_module MMF_GR_Add \
  --llm_model_fusion GPT2
```

---

## Dataset

**EPA-Air** from the [TIME-IMM benchmark](https://github.com/blacksnail789521/Time-IMM) (NeurIPS 2025).

| Property | Value |
|---|---|
| Entities (counties) | 8 |
| Numerical features | 4 (AQI, Ozone, PM2.5, Temperature) |
| Total observations | 49,552 |
| Unique timestamps | 6,587 |
| Mean inter-observation interval | 1.02 hours |
| Feature Observability Entropy | 0.3777 (lowest in benchmark) |
| Text entries | 1,244 weather news summaries |
| Context window | 7 days |
| Forecast horizon | 7 days |
| Train / Val / Test split | 60% / 20% / 20% (chronological) |
| Irregularity type | Multi-Source Asynchrony (Artifact-Based) |

---

## Differences from Paper Protocol

Two intentional deviations from the official `main_all.py` configuration, documented for transparency:

1. **Text encoder:** We use GPT-2 (768-dim). The paper's default is DeepSeek, which requires 24GB+ VRAM for the LLM alone. GPT-2 is a practical choice for a class project on 40GB A100.
2. **TTF/MMF grid:** We use `TTF_RecAvg + MMF_GR_Add` only. The paper iterates over all 4 combinations (`TTF_RecAvg`/`TTF_T2V_XAttn` × `MMF_GR_Add`/`MMF_XAttn_Add`) and reports the best. Our "multi" numbers may be slightly worse than the paper's for models where a different combination would have been optimal.

Both deviations apply uniformly across all model runs, so within-experiment comparisons remain valid.

---

## References

1. Chang, C., Hwang, J., Shi, Y., Wang, H., Wang, W., Peng, W.-C., & Chen, T.-F. (2025). TIME-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series. *NeurIPS 2025 Datasets & Benchmarks Track*. [arXiv:2506.10412](https://arxiv.org/abs/2506.10412)
2. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*.
3. Nie, Y., Nguyen, N.H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*.
4. Wu, H., Xu, J., Wang, J., & Long, M. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. *ICLR 2023*.
5. Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y., & Zhou, J. (2024). TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. *ICLR 2024*.
6. Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J.Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., & Wen, Q. (2024). Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. *ICLR 2024*.
7. Zhang, W., et al. (2024). tPatchGNN: Temporal Patch Graph Neural Networks for Time Series Forecasting.
8. Schirmer, M., Eltayeb, M., Lessmann, S., & Rudolph, M. (2022). Modeling Irregular Time Series with Continuous Recurrent Units. *ICML 2022*.
9. Rubanova, Y., Chen, R.T.Q., & Duvenaud, D. (2019). Latent ODEs for Irregularly-Sampled Time Series. *NeurIPS 2019*.
10. Biloš, M., Sommer, J., Rangapuram, S.S., Januschowski, T., & Günnemann, S. (2021). Neural Flows: Efficient Alternative to Neural ODEs. *NeurIPS 2021*.

---

## License

This project is for academic purposes only (STAT 8240 course project). The IMM-TSF benchmark library is licensed under MIT. The TIME-IMM dataset is available under the terms specified by the authors.
