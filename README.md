# Replicating and Extending IMM-TSF on EPA-Air Quality Data

**STAT 8240 — Data Mining II | Spring 2026**
**Kennesaw State University**

---

## Project Overview

This project replicates and extends the **IMM-TSF** benchmark framework (Chang et al., NeurIPS 2025 Datasets & Benchmarks Track) on the **EPA-Air** dataset — one of nine datasets in the TIME-IMM collection. EPA-Air represents the *Multi-Source Asynchrony* irregularity type, where four environmental sensors (AQI, Ozone, PM2.5, Temperature) report at independent, asynchronous schedules across eight U.S. counties.

We successfully reproduced 7 of the 11 baseline models from the paper's Table 11, attempted the remaining 4 (documenting their failure modes), and conducted three targeted ablation experiments that probe the paper's architectural claims about text encoder choice, architecture-family effects, and whether multimodal gains reflect genuine semantic signal or regularization artifacts.

---

## Key Results

### Reproduction (7 of 11 models)

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

### Remaining 4 Models (attempted)

| Model | Status | Failure Mode |
|---|---|---|
| TimeLLM | Partial success | Default patch_size=24 incompatible with EPA-Air's sparse features; resolved with reduced patch size |
| CRU | Failed | ODE solver instability on EPA-Air's 1.02-hour inter-observation interval over a 7-day horizon |
| LatentODE | Failed | Gradient explosion from accumulated numerical errors in continuous-time integration |
| NeuralFlow | Failed | Same ODE solver instability as CRU/LatentODE |

### Extension Findings

1. **Encoder choice matters on EPA-Air** — switching from GPT-2 to BERT produced performance swings of up to 13.3 percentage points for individual models (TTM: +6.0% → −7.3%), directly challenging the paper's Figure 6c claim that encoder choice is negligible.
2. **Architecture family predicts text benefit** — non-patch models (DLinear, Informer, TimesNet, TimeMixer) benefit from text with both encoders (avg −3%), while patch-based models (PatchTST, TTM, tPatchGNN) are harmed by GPT-2 (+4.6%) but helped by BERT (−3.0%).
3. **Placebo test** — only 3 of 7 models show genuine semantic benefit from real text. Two models show a placebo effect (noise performs within 1% of real text), and two are actively harmed by real text more than by noise.

---

## Repository Structure

```
├── README.md
├── DM-ll_PROJECT_REPORT_.pdf                  # Final project report (19 pages)
│
├── notebooks/
│   ├── DM-ll_PROJECT_CODE.ipynb               # 7-model reproduction + 3 ablation experiments
│   │                                          #   (encoder swap, architecture family analysis,
│   │                                          #    placebo test), results tables, visualizations
│   │
│   ├── Extention.ipynb                        # Per-county evaluation extension
│   │                                          #   Spearman correlation + cluster-bootstrap CI
│   │                                          #   across 8 counties × 7 models
│   │
│   ├── DM_ll_Project_7_Models.ipynb           # Independent 7-model replication + ablations
│   │
│   └── DM_ll_Project_4_Remaining_Models.ipynb # TimeLLM, CRU, LatentODE, NeuralFlow
│                                              #   attempts with failure documentation
│
└── results/                                   # (persisted on Google Drive)
    └── runs.jsonl                             # All logged experiment results
```

---

## Reproducing the Experiments

### Prerequisites

- Google Colab Pro with NVIDIA A100 GPU (40GB VRAM)
- Kaggle account (for dataset download)
- Google Drive (for embedding cache persistence across sessions)

### Setup

1. Open any notebook in Google Colab (`DM-ll_PROJECT_CODE.ipynb` is the most comprehensive).
2. Run the setup cells — they mount Google Drive, restore Kaggle credentials, clone the official [IMM-TSF](https://github.com/blacksnail789521/IMM-TSF) and [Time-IMM](https://github.com/blacksnail789521/Time-IMM) repositories, install PyTorch 2.7.0 + dependencies (`reformer_pytorch`, `stribor`, `geotorch`), download the EPA-Air dataset from Kaggle, and restore pre-computed GPT-2 text embeddings from the Drive cache.

### Running Models

All models use the same CLI pattern via the official `main.py`:

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

The 11 models from the paper's Table 11: `Informer`, `DLinear`, `PatchTST`, `TimesNet`, `TimeMixer`, `TimeLLM`, `TTM`, `CRU`, `LatentODE`, `NeuralFlow`, `tPatchGNN`.

For `tPatchGNN`, add `--patch_size 4` (default 24 is incompatible with EPA-Air's sparse features).

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
| Context window / Forecast horizon | 7 days / 7 days |
| Train / Val / Test split | 60% / 20% / 20% (chronological) |
| Irregularity type | Multi-Source Asynchrony (Artifact-Based) |

---

## Differences from Paper Protocol

Two intentional deviations from the official `main_all.py` configuration, documented for transparency:

1. **Text encoder:** We use GPT-2 (768-dim). The paper's default is DeepSeek, which requires 24GB+ VRAM for the LLM alone. GPT-2 is a practical choice on a 40GB A100.
2. **TTF/MMF grid:** We use `TTF_RecAvg + MMF_GR_Add` only. The paper iterates over all 4 combinations (`TTF_RecAvg`/`TTF_T2V_XAttn` × `MMF_GR_Add`/`MMF_XAttn_Add`) and reports the best.

Both deviations apply uniformly across all runs, so within-experiment comparisons remain valid.

---

## References

1. Chang, C., et al. (2025). TIME-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series. *NeurIPS 2025*. [arXiv:2506.10412](https://arxiv.org/abs/2506.10412)
2. Zeng, A., et al. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*.
3. Nie, Y., et al. (2023). A Time Series is Worth 64 Words. *ICLR 2023*.
4. Wu, H., et al. (2023). TimesNet: Temporal 2D-Variation Modeling. *ICLR 2023*.
5. Wang, S., et al. (2024). TimeMixer: Decomposable Multiscale Mixing. *ICLR 2024*.
6. Jin, M., et al. (2024). Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. *ICLR 2024*.
7. Zhang, W., et al. (2024). tPatchGNN: Temporal Patch Graph Neural Networks.
8. Schirmer, M., et al. (2022). Modeling Irregular Time Series with Continuous Recurrent Units. *ICML 2022*.
9. Rubanova, Y., et al. (2019). Latent ODEs for Irregularly-Sampled Time Series. *NeurIPS 2019*.
10. Biloš, M., et al. (2021). Neural Flows: Efficient Alternative to Neural ODEs. *NeurIPS 2021*.

---

## License

Academic use only (STAT 8240 course project). The IMM-TSF benchmark library is licensed under MIT. The TIME-IMM dataset is available under the terms specified by the original authors.
