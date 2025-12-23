[中文](./README(CH).md#核心特性) | English
# 🧬 GP-Alpha-Miner: GPU-Accelerated Genetic Programming Framework

**GP-Alpha-Miner** is a high-performance, industrial-grade quantitative factor mining framework. It combines the searching capability of Genetic Programming (GP) with the parallel computing power of GPUs, aimed at automatically discovering Alpha factors with high ICIR and low turnover rates within massive datasets.

This project is specifically designed for the **A-share market**, featuring built-in rigorous **anti-overfitting** and **anti-look-ahead bias** mechanisms. It is suitable for quantitative researchers, strategy developers, and financial engineering students.

---

## ✨ Key Features

* 🚀 **GPU Ultra-Fast Computation**: Underlying operators are implemented based on `CuPy` matrix operations, supporting second-level factor calculation for 5000+ stocks across the entire market, with efficiency improved by 50-100x compared to CPUs.
* 🛡️ **Rigorous Data Alignment**:
* **Feature De-dimensioning**: Completely masks absolute prices (such as Open, Close), forcing the use of relative returns and ratios to prevent the machine from mining pseudo-factors.
* **Target Shift**: Strictly executes `Target = Returns.shift(-1)`, ensuring that T-day data is used to predict T+1 day returns, eliminating look-ahead bias.


* 🧠 **Intelligent Fitness Evaluation**:
* **IC_METHOD = 'Rank'**: Forces the use of Spearman rank correlation to resist data noise.
* **Multiple Penalties**: Introduces Turnover Penalty and Parsimony Pressure (complexity penalty) to guide the machine toward mining factors with concise logic and stable holdings.


* 🧩 **Rich Operator Library**: Built-in WorldQuant Alpha101-style Time-Series and Cross-Sectional operators.
* 💾 **Automated Pipeline**: Supports Grid Search, automatic result archiving, and automatic merging of factor wide tables.

---

## 🏗️ Architecture Overview

```text
GP-Alpha-Miner/
├── config.py            # 🎛️ [Center] Global parameter configuration (GPU toggle, population size, penalty coefficients, etc.)
├── data_loader.py       # 🏗️ [Foundation] Data cleaning, feature engineering, CPU->GPU memory transfer
├── operators.py         # 🧮 [Engine] CuPy-based vectorized operator library (all running on GPU)
├── fitness.py           # ⚖️ [Referee] Fitness function, numerical cleaning, and scoring logic
├── run.py               # 🚀 [Launch] DEAP evolutionary algorithm main loop
├── utils.py             # 💾 [Logistics] Result saving, factor merging, and logging
└── README.md            # 📖 Instruction manual

```

---

## 📊 Input Features

To prevent the GP algorithm from "cheating" by fitting absolute stock price values (e.g., Kweichow Moutai at 2000 RMB vs. low-priced stocks at 2 RMB), this framework performs mandatory feature engineering in `data_loader.py`. The GP can only see the following **18 relative indicators**, categorized into basic, high-order statistical, and volatility types:

| Code (Arg Name) | Full Name | Mathematical Definition (Formula) | Physical Meaning |
| --- | --- | --- | --- |
| **RET** | Return |  | **Daily Return**: The most basic source for momentum/reversal signals. |
| **GAP** | Open Gap |  | **Opening Gap**: Reflects the impact of overnight information on stock prices. |
| **HL_R** | High-Low Ratio |  | **Intraday Amplitude**: Reflects the degree of long-short divergence during the day. |
| **CO_R** | Close-Open Ratio |  | **K-line Entity Growth**: Reflects the active push intent of intraday capital. |
| **L_VOL** | Log Volume |  | **Log Volume**: Reflects transaction activity (dimensionless). |
| **TO_RATE** | Turnover Rate |  | **Turnover Rate**: Reflects the heat of chip exchange. |
| **L_CAP** | Log Market Cap |  | **Log Market Cap**: Used to mine market cap preferences (large-cap/small-cap). |
| **VWAP_D** | VWAP Distance |  | **Average Price Deviation**: Position of closing price relative to the full-day average price; a strong reversal signal. |
| **AMI** | Amihud Illiquidity |  | **Illiquidity Indicator**: Price fluctuation per unit of trading volume, capturing liquidity premiums. |
| **BODY** | Body Ratio |  | **Body Ratio**: Proportion of the K-line body relative to the full-day amplitude, used to judge trend strength. |
| **UP_S** | Upper Shadow |  | **Upper Shadow Rate**: Reflects the strength of overhead selling pressure. |
| **LO_S** | Lower Shadow |  | **Lower Shadow Rate**: Reflects the strength of underlying support. |
| **LOG_RET** | Log Return |  | **Log Return**: Compared to simple returns, it better fits normal distribution assumptions and is suitable for capturing non-linear features. |
| **SKEW** | Return Skewness |  | **Return Skewness**: Measures distribution asymmetry (left/right skew), capturing tail risks. |
| **KURT** | Return Kurtosis |  | **Return Kurtosis**: Measures the "fat-tail" degree of the distribution, capturing the probability of extreme market conditions. |
| **BB_W** | Bollinger Width |  | **Bollinger Width**: A classic volatility indicator reflecting market compression (accumulation) and expansion (breakout). |
| **ATR** | Normalized ATR |  | **Normalized ATR**: De-dimensioned average true range, measuring the absolute intensity of price fluctuations. |
| **V_SKEW** | Volatility Skew |  | **Volatility Skew**: Measures the "leverage effect." Negative values usually imply that price drops lead to amplified volatility (panic). |

> **Notes**:
> 1. All calculations involving division include a `1e-6` epsilon protection to prevent division-by-zero errors.
> 2. High-order statistics (Skew, Kurt, Vol_Skew) are typically calculated based on a window of the past 20 trading days (approximately one month).
> 
> 

---

## 🧮 Operator Library

This framework implements the following GPU operators in `operators.py`. All operators support full-matrix parallel computation.

### 1. Basic Math Operators

| Operator | Description | Arity (Number of Arguments) |
| --- | --- | --- |
| `add(x, y)` |  | 2 |
| `sub(x, y)` |  | 2 |
| `mul(x, y)` |  | 2 |
| `protected_div(x, y)` |  (Returns 1 if ) | 2 |
| `abs_val(x)` |  | 1 |
| `log_abs(x)` |  | 1 |
| `sqrt_abs(x)` |  | 1 |

### 2. Time-Series Operators (Rolling Window)

Used to extract features over time series. Window length  is typically fixed at 5, 10, or 20.

| Operator | Description | Logic |
| --- | --- | --- |
| `ts_mean(x, d)` | Rolling Mean |  |
| `ts_std(x, d)` | Rolling Std Dev |  |
| `ts_delta(x, d)` | Time-Series Difference |  |
| `ts_max(x, d)` | Rolling Max |  |
| `ts_min(x, d)` | Rolling Min |  |
| `ts_rank(x, d)` | Time-Series Rank | Percentile rank of  within the past  days of data (0~1) |
| `ts_corr(x, y, d)` | Rolling Correlation |  |
| `decay_linear(x, d)` | Linear Decay | Weighted average, where more recent data has higher weight () |

### 3. Cross-Sectional Operators

Used to compare the relative strength of different stocks at the same point in time.

| Operator | Description | Logic |
| --- | --- | --- |
| `cs_rank(x)` | Cross-Sectional Rank | Transforms intraday factor values into a percentile rank of 0~1. |
| `cs_scale(x)` | Cross-Sectional Z-Score | . De-dimensioning allows data from different distributions to be added or subtracted. |

---

## ⚙️ Quick Start

### 1. Environment Installation

It is recommended to use Conda to manage the environment.

```bash
# 1. Create environment
conda create -n gp_miner python=3.8
conda activate gp_miner

# 2. Install CuPy with CUDA support (Please check your CUDA version via nvidia-smi first)
# For CUDA 12.x:
pip install cupy-cuda12x
# For CUDA 11.x:
pip install cupy-cuda11x

# 3. Install other dependencies
pip install pandas numpy deap pyarrow

```

### 2. Data Preparation

Please place your data in the `data/` directory:

* `data/data.parquet`: Contains fields such as `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`, `TurnOverValue`, etc.
* `data/ret_df.parquet`: Contains `Date`, `Ticker`, `Target_Return`.

### 3. Running the Miner

```bash
python run.py

```

### 4. Results Output

After the program finishes, results will be saved in an automatically generated experiment folder under `GP/GPFactors/`.

* **`formulas.csv`**: Mined factor formulas and fitness scores.
* **`All_Factors_Merged.parquet`**: Wide table data of all Top factors (automatically merged, ready for machine learning).

---

## 🔬 Advanced: Tuning Guide

Modify parameters in `config.py` to control the mining direction:

* **Want to mine low-frequency/fundamental factors?**
* Set `IC_METHOD = 'rank'`
* Increase `PENALTY_TURNOVER = 0.1` (Strictly penalize high turnover)
* Increase `GENERATIONS = 30` (Deep search)


* **Want to mine high-frequency/price-volume factors?**
* Set `IC_METHOD = 'rank'`
* Decrease `PENALTY_TURNOVER = 0.01` (Allow some turnover)
* Focus on `ts_corr`, `ts_delta` type operators


* **Found factors are all `CO_R` (Overfitting)?**
* Increase `PENALTY_COMPLEXITY`
* Or temporarily mask `CO_R` input in `run.py`



---

## ⚠️ FAQ

**Q: Why are the generated factors all -999 points?**
A: Usually, this is because the number of parameters in `pset` in `run.py` does not match the number of inputs in `fitness.py`. Please check if `pset = gp.PrimitiveSet("MAIN", 12)` is correctly set to 12.

**Q: Why can't I see GPU usage?**
A: GP calculation is impulsive. The GPU will only be fully loaded during the Evaluation phase (at the start of each generation); the Crossover and Mutation phases are handled by the CPU. You can observe this using `watch -n 0.5 nvidia-smi`.

**Q: How can I confirm that no look-ahead bias was used?**
A: Check `data_loader.py`. We have built-in `target_shifted = target_raw.shift(-1)`, which ensures that Row(T) features correspond to Row(T+1) returns.

---

## 🤝 Contribution

Feel free to submit an Issue or Pull Request to add new operators (such as complex logic from Alpha191) or optimize computational efficiency.