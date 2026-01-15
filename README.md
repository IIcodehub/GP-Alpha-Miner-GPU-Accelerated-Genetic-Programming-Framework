[简体中文](./README(CH).md) | English

# 🧬 GP-Alpha-Miner: GPU-Accelerated Genetic Programming Framework

![CuPy](https://img.shields.io/badge/Backend-CuPy%20(GPU)-green)
![DEAP](https://img.shields.io/badge/Framework-DEAP-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**GP-Alpha-Miner** is a high-performance quantitative factor automatic mining framework designed specifically for the A-share market. By combining the global search capabilities of Genetic Programming (GP) with the massive parallel computing power of **GPU (CuPy)**, this project achieves second-level factor backtesting and evolution. It features deep financial production logic: from non-linear volatility features to structure-constrained mutation and automated grid search.

---

## ✨ Key Highlights

* 🚀 **Extreme Performance**: The operator library is fully vectorized and operates directly in CuPy device memory. On a $5000 \times 1000$ data matrix, the iteration efficiency is approximately 100x higher than traditional CPU-based frameworks.

* ⚙️ **Automated Grid Search**: Built-in parameter combination traversal module that automatically optimizes hyperparameters such as population size, generations, and turnover penalties, archiving results for each experiment independently.
* 🧬 **Enhanced Evolution Logic**:
    * **Warm Start**: Introduces a prior seed library to inject classic financial logic into the population's genes.
    * **Structure Constrained**: Provides isomorphic crossover and point mutation to ensure the "logical skeleton" remains intact.
* 🛡️ **Industrial-Grade Anti-Overfitting**:
    * **Feature De-dimensioning**: Masks absolute prices and forces the use of 18 relative features like returns and bias ratios.
    * **Look-ahead Prevention**: Strictly executes `shift(-1)` on target returns to ensure all signals are based on known information.
---

## 🏗️ System Architecture

```text
GP-Alpha-Miner/
├── config.py           # 🎛️ [Control Tower] Global parameter definitions (population, penalties, switches)
├── data_loader.py      # 🏗️ [Bedrock] 18D advanced feature engineering, async GPU memory loading
├── operators.py        # 🧮 [Engine] 30+ CuPy-based vectorized operators (Time-series/Cross-sectional/Regression)
├── fitness.py          # ⚖️ [Judge] RankIC evaluation, turnover proxy, complexity penalty
├── gp_logic.py         # 🧬 [Evolution] Structured mutation/crossover, seed loader
├── seeds.py            # 🌱 [Seed Bank] Pre-set expert logic formulas
├── run.py              # 🚀 [Main] DEAP environment registration and mining loop
├── grid_search.py      # 🔍 [Optimization] Automated parameter grid traversal module
└── utils.py            # 💾 [Logistics] GPU data transfer, factor storage, auto-merge wide tables
```

---

## 📊 18-Dimensional Input Features

The system fixes leaf nodes to 18 engineered features (`ARG0` to `ARG17`), ensuring the mined logic maintains economic interpretability.

### 1. Basic Price-Volume Features
| No. | Identifier | Mathematical Definition (LaTeX) | Physical Meaning |
| :--- | :--- | :--- | :--- |
| `ARG0` | `RET` | $R_t = \frac{P_{close,t}}{P_{close,t-1}} - 1$ | Daily Return (Momentum/Reversal) |
| `ARG1` | `OPEN_GAP` | $\frac{O_t}{P_{close,t-1}} - 1$ | Overnight Gap (Opening momentum) |
| `ARG2` | `HL_RATIO` | $\frac{H_t}{L_t} - 1$ | Intraday Range (Long-short divergence) |
| `ARG3` | `CO_RATIO` | $\frac{P_{close,t}}{O_t} - 1$ | Intraday Body Return (Bullish strength) |
| `ARG4` | `LOG_VOL` | $\ln(V_t + 1)$ | Trading Activity (Log volume) |
| `ARG5` | `TO_RATE` | $\frac{V_t}{\text{FreeShares}}$ | Turnover Rate (Liquidity intensity) |
| `ARG6` | `LOG_CAP` | $\ln(\text{FloatCap}_t)$ | Market Cap (Log scale) |

### 2. Structure, VWAP & Liquidity
| No. | Identifier | Mathematical Definition (LaTeX) | Physical Meaning |
| :--- | :--- | :--- | :--- |
| `ARG7` | `VWAP_D` | $\frac{P_{close,t}}{\text{VWAP}_t} - 1$ | VWAP Bias (Cost deviation) |
| `ARG8` | `AMIHUD` | $\frac{|R_t|}{\text{Amount}_t} \times 10^9$ | Amihud Illiquidity (Impact cost) |
| `ARG9` | `BODY_R` | $\frac{|P_t - O_t|}{H_t - L_t + \epsilon}$ | K-line Body Ratio (Trend reliability) |
| `ARG10` | `UP_SHD` | $\frac{H_t - \max(O, P)}{H_t - L_t + \epsilon}$ | Upper Shadow (Resistance/Selling pressure) |
| `ARG11` | `LO_SHD` | $\frac{\min(O, P) - L_t}{H_t - L_t + \epsilon}$ | Lower Shadow (Support/Buying pressure) |

### 3. High-Order Non-linear Statistics
| No. | Identifier | Mathematical Definition (LaTeX) | Physical Meaning |
| :--- | :--- | :--- | :--- |
| `ARG12` | `LOG_RET` | $\ln(P_t / P_{t-1})$ | Log Return (Non-linear changes) |
| `ARG13` | `SKEW` | $\text{Skew}(R, 20)$ | Return Skewness (Distribution asymmetry) |
| `ARG14` | `KURT` | $\text{Kurt}(R, 20)$ | Return Kurtosis (Extreme event frequency) |
| `ARG15` | `BB_WIDTH` | $\frac{Upper_{20} - Lower_{20}}{Mid_{20}}$ | Bollinger Band Width (Volatility squeeze) |
| `ARG16` | `ATR` | $\frac{\text{ATR}_{14}}{P_t}$ | Normalized Average True Range |
| `ARG17` | `VOL_SKEW` | $\text{Corr}(R, \vert R \vert, 20)$ | Volatility Skew |

---

## 🧮 Operator Library Manual

All operators perform lookbacks over a window $d$ at time $t$.

### 1. Basic Math & Non-Linear Operators
| Operator | Description | Arity | Logic/Formula |
| :--- | :--- | :--- | :--- |
| `add(x, y)` | Addition | 2 | $x + y$ |
| `sub(x, y)` | Subtraction | 2 | $x - y$ |
| `mul(x, y)` | Multiplication | 2 | $x \times y$ |
| `protected_div(x, y)` | Protected Div | 2 | $x / y$ (Returns 0 or 1 if $y \approx 0$) |
| `abs_val(x)` | Absolute Value | 1 | $\lvert x \rvert$ |
| `sigmoid(x)` | Sigmoid | 1 | $1 / (1 + e^{-x})$ (Map to (0,1)) |
| `signed_power(x, a)` | Signed Power | 1 | $sign(x) \times \lvert x \rvert^a$ |

### 2. Rolling Statistics Operators
| Operator | Description | Logic |
| :--- | :--- | :--- |
| `ts_mean(x, d)` | Rolling Mean | $\mu = \frac{1}{d} \sum x_i$ |
| `ts_std(x, d)` | Rolling Std | $\sigma = \sqrt{Var(x)}$ |
| `ts_delta(x, d)` | Time Difference | $x_t - x_{t-d}$ |
| `ts_rank(x, d)` | Rolling Rank | Percentile rank of $x_t$ over $d$ days (0~1) |
| `ts_decay_linear(x, d)`| Linear Decay | Weighted average ($w_t=d, w_{t-1}=d-1...$) |

### 3. Advanced Trend & Distribution
| Operator | Description | Logic |
| :--- | :--- | :--- |
| `ts_zscore(x, d)` | Rolling Z-Score | $(x_t - \mu_d) / \sigma_d$ (Mean reversion) |
| `ts_skew(x, d)` | Rolling Skew | 3rd moment (Asymmetry) |
| `ts_tslope(x, d)` | Time Slope | Regression slope of $x$ against time $t$ |
| `ts_rsi(x, d)` | RSI | Standard Relative Strength Index |

### 4. Pairwise & Regression Operators
| Operator | Description | Arity | Logic |
| :--- | :--- | :--- | :--- |
| `ts_corr(x, y, d)` | Rolling Corr | 2 | $Cov(x,y) / (\sigma_x \sigma_y)$ |
| `ts_resid(x, y, d)` | Rolling Resid | 2 | $y - (\alpha + \beta x)$ (Alpha idiosyncratic) |

---

## ⚖️ Fitness & Grid Search

### 1. Turnover Penalty Proxy
To balance calculation speed with execution costs, we calculate the **cross-sectional auto-correlation**:
$$\rho_{\text{auto}} = \text{CosineSimilarity}(F_t, F_{t+1})$$
$$\text{Penalty}_{\text{turnover}} = (1 - \text{Avg}(\rho_{\text{auto}})) \times \lambda_{\text{turn}}$$
*Translation: Higher auto-correlation leads to lower turnover and lower penalties.*

### 2. Final Fitness Formula
$$\text{Fitness} = \text{ICIR} - \text{Penalty}_{\text{turnover}} - \text{Length}(\text{Formula}) \times \lambda_{\text{comp}}$$
* **ICIR**: $\frac{\text{Mean}(IC)}{\text{Std}(IC)}$, defaults to **Rank IC**.
* **Direction Penalty**: If $\text{Mean}(IC) < 0$, the score is heavily penalized (0.1x).

---

## 🧬 Evolutionary Mechanisms


### 1. Warm Start
The system initializes with a probability of loading manual logical skeletons from `seeds.py` (e.g., `ts_corr_10(RET, L_VOL)`).

### 2. Structure Constrained
* **Point Mutation**: Only replaces leaf nodes, keeping the parent operators intact.
* **Isomorphic Crossover**: Only swaps subtrees with similar depth and structure to preserve logical "DNA."

---

## 🚀 Quick Start

1.  **Install Environment**:
    ```bash
    pip install pandas numpy cupy-cuda11x deap pyarrow
    ```
2.  **Run Grid Search**:
    ```bash
    # Run after configuring grid_search.py
    python grid_search.py
    ```
3.  **Outputs**:
    Results are stored in `GP/GPFactorsRound/`.
    * `formulas.csv`: List of Top factors with math expressions and scores.
    * `All_Factors_Merged.parquet`: Merged wide table with `TradingDay` and `SecuCode` indices.