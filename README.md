
# Stock Price Prediction using LSTM and GRU Neural Networks

A comparative study of recurrent neural network architectures for next-day stock price prediction, using Apple Inc. (AAPL) daily historical data from 1980 to 2020.

This repository contains the complete code, notebook, and documentation for an MSc Data Science coursework submission at Arden University (AI and Neural Networks module).

---

## Project Overview

This project investigates whether LSTM and GRU recurrent neural networks can accurately predict the next day's closing price for Apple Inc. stock. The study uses a controlled experimental grid of eight model configurations to isolate the effect of each hyperparameter, and compares the best neural model against a trivial persistence baseline.

### Key research questions

- Which recurrent architecture (LSTM or GRU) performs better for next-day price prediction?
- How do sequence length, number of units, and dropout rate individually affect prediction accuracy?
- Can a neural network outperform a simple persistence baseline (tomorrow's price equals today's)?

---

## Dataset

- **Source:** Apple Inc. (NASDAQ ticker: AAPL), provided as part of Arden University module coursework
- **Format:** Daily OHLCV (Open, High, Low, Close, Volume) plus Adjusted Close
- **Period:** 12 December 1980 to 1 April 2020
- **Records:** 9,909 trading days
- **Modelling window:** Restricted to 2010 onwards to avoid training across incompatible price regimes

---

## Methodology

### Data preparation

- Chronological train / validation / test split (70% / 15% / 15%)
- MinMax scaling fitted on training data only (no leakage)
- Sliding-window sequences of 30 or 60 days

### Controlled experimental grid

Eight configurations, each differing from at least one other by a single hyperparameter:

| Model | Cell | Sequence | Units | Dropout |
|-------|------|----------|-------|---------|
| M1 | LSTM | 30 | 50 | 0.2 |
| M2 | LSTM | 30 | 50 | 0.3 |
| M3 | LSTM | 30 | 64 | 0.2 |
| M4 | LSTM | 60 | 50 | 0.2 |
| M5 | LSTM | 60 | 64 | 0.3 |
| M6 | GRU | 30 | 50 | 0.2 |
| M7 | GRU | 60 | 50 | 0.2 |
| M8 | GRU | 60 | 64 | 0.3 |

### Training setup

- Two stacked recurrent layers with dropout
- Adam optimiser (default base learning rate 0.001)
- MSE loss function
- Early stopping on validation loss with patience 4 and restore_best_weights
- Maximum 40 epochs, batch size 128
- Random seed fixed at 42 for reproducibility

### Evaluation

- Model selection on the **validation set**
- Test set used **once** at the end on the selected model only
- Metrics reported: RMSE, MAE, MAPE, R²
- Persistence baseline (tomorrow's Close = today's Close) used as reference

---

## Key Findings

- **Selected model:** M8 (60-day GRU with 64 units and 0.3 dropout) achieved the best validation performance
- **M8 test-set performance:** RMSE 18.93, MAE 13.39, MAPE 5.33%, R² 0.829
- **Persistence baseline:** RMSE 5.65, MAE 3.55, MAPE 1.59%, R² 0.985
- **Important finding:** The persistence baseline outperforms every neural network on every metric, consistent with the well-documented random-walk nature of daily stock prices. This motivates the future-work extensions (technical indicators, exogenous variables, alternative prediction targets).

---

## Repository Structure

```
.
├── README.md                          # This file
├── Stock_Prediction_Colab.ipynb       # Complete reproducible notebook
├── Stock_Price_Data_3921.csv          # Dataset (Apple OHLCV)
├── report/
│   └── Stock_Price_Data_3921.docx     # Full written report
├── figures/
│   ├── figure_1_price_trend.png
│   ├── figure_2_volume.png
│   ├── figure_3_model_comparison.png
│   ├── figure_4_loss_curves.png
│   └── figure_5_actual_vs_pred.png
└── results/
    └── run_results.json               # Numeric outputs from the run
```

---

## How to Run

### Option 1: Google Colab (recommended)

1. Open `Stock_Prediction_Colab.ipynb` in Google Colab
2. Upload `Stock_Price_Data_3921.csv` when Cell 2 prompts for it
3. Run all cells (`Runtime → Run all`)
4. Expected runtime: 3–5 minutes on CPU, ~2 minutes on GPU

### Option 2: Local Python

Requirements:

```
python >= 3.11
tensorflow >= 2.21
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
```

Install dependencies and run:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
python pipeline.py
```

---

## Software Versions

Reproducibility is critical for this work. The reported results were generated with:

- Python 3.11
- TensorFlow 2.21.0
- scikit-learn 1.3
- pandas 2.0
- NumPy 1.24
- matplotlib 3.7
- Random seed: 42 (set for `random`, NumPy, and TensorFlow)

---

## References

Key academic sources used in this project:

- Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP*.
- Fischer, T., & Krauss, C. (2018). Deep learning with LSTM networks for financial market predictions. *European Journal of Operational Research*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
- Selvin, S., et al. (2017). Stock price prediction using LSTM, RNN and CNN-sliding window model. *ICACCI*.
- Siami-Namini, S., et al. (2018). A comparison of ARIMA and LSTM in forecasting time series. *ICMLA*.
- Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*.

## Author

**Khalil Animashaun (2026)**
