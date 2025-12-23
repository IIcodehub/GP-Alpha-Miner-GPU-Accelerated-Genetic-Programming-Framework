
简体中文 | [English](./README.md)
# 🧬 GP-Alpha-Miner: GPU-Accelerated Genetic Programming Framework

[](https://www.python.org/)
[](https://cupy.dev/)
[](https://github.com/DEAP/deap)
[](https://opensource.org/licenses/MIT)

**GP-Alpha-Miner** 是一个高性能、工业级的量化因子挖掘框架。它结合了遗传规划（Genetic Programming, GP）的搜索能力与 GPU 的并行计算能力，旨在海量数据中自动发现具有高ICIR和低换手率的 Alpha 因子。

本项目特别针对**A股市场**设计，内置了严格的**防过拟合**和**防未来函数**机制，适合量化研究员、策略开发者及金融工程学生使用。

-----

## ✨ 核心特性 (Key Features)

  * 🚀 **GPU 极速计算**: 底层算子基于 `CuPy` 矩阵运算实现，支持全市场 5000+ 股票的秒级因子计算，效率较 CPU 提升 50-100 倍。
  * 🛡️ **严格的数据对齐**:
      * **特征去量纲化**: 彻底屏蔽绝对价格（如 Open, Close），强制使用相对收益率和比率，防止机器挖掘出伪因子。
      * **Target Shift**: 严格执行 `Target = Returns.shift(-1)`，确保使用 T 日数据预测 T+1 日收益，杜绝未来函数。
  * 🧠 **智能适应度评估**:
      * **IC\_METHOD = 'Rank'**: 强制使用 Spearman 排序相关性，抵抗数据噪音。
      * **多重惩罚**: 引入换手率惩罚（Turnover Penalty）和复杂度惩罚（Parsimony Pressure），引导机器挖掘逻辑简洁、持仓稳定的因子。
  * 🧩 **丰富的算子库**: 内置 WorldQuant Alpha101 风格的时序（Time-Series）与截面（Cross-Sectional）算子。
  * 💾 **自动化流水线**: 支持网格搜索（Grid Search）、结果自动归档、因子宽表自动合并。

-----

## 🏗️ 架构概览 (Architecture)

```text
GP-Alpha-Miner/
├── config.py           # 🎛️ [中枢] 全局参数配置（显卡开关、种群大小、惩罚系数等）
├── data_loader.py      # 🏗️ [基石] 数据清洗、特征工程、CPU->GPU 显存搬运
├── operators.py        # 🧮 [引擎] 基于 CuPy 的向量化算子库（全部运行在 GPU）
├── fitness.py          # ⚖️ [裁判] 适应度函数、数值清洗与评分逻辑
├── run.py              # 🚀 [启动] DEAP 进化算法主循环
├── utils.py            # 💾 [后勤] 结果保存、因子合并与日志记录
└── README.md           # 📖 说明文档
```

-----

## 📊 基础输入特征 (Input Features)

为了防止 GP 算法“偷懒”去拟合股价绝对值（如 2000元的茅台和 2元的低价股），本框架在 `data_loader.py` 中进行了强制特征工程。GP 只能看到以下 **18 个相对指标**，分为基础类、高阶统计类和波动率类：

| 代码 (Arg Name) | 全称 | 数学定义 (Formula) | 物理含义 |
| :--- | :--- | :--- | :--- |
| **RET** | Return | $Close_t / Close_{t-1} - 1$ | **日收益率**：最基础的动量/反转信号源。 |
| **GAP** | Open Gap | $Open_t / Close_{t-1} - 1$ | **开盘跳空**：反映隔夜信息对股价的冲击。 |
| **HL_R** | High-Low Ratio | $High_t / Low_t - 1$ | **日内振幅**：反映当天的多空分歧程度。 |
| **CO_R** | Close-Open Ratio | $Close_t / Open_t - 1$ | **K线实体涨幅**：反映日内资金的主动推升意愿。 |
| **L_VOL** | Log Volume | $\ln(Volume_t + 1)$ | **对数成交量**：反映成交活跃度（去量级）。 |
| **TO_RATE** | Turnover Rate | $Volume_t / Shares\_Outstanding$ | **换手率**：反映筹码交换的热度。 |
| **L_CAP** | Log Market Cap | $\ln(MarketCap_t + 1)$ | **对数市值**：用于挖掘市值偏好（大盘/小盘）。 |
| **VWAP_D** | VWAP Distance | $Close_t / VWAP_t - 1$ | **均价偏离度**：收盘价相对于全天均价的位置，强反转信号。 |
| **AMI** | Amihud Illiquidity | $\lvert RET_t \rvert / Amount_t$ | **非流动性指标**：单位成交额带来的价格波动，捕捉流动性溢价。 |
| **BODY** | Body Ratio | $\lvert Close_t - Open_t \rvert / (High_t - Low_t)$ | **实体率**：K线实体占全天振幅的比例，判断趋势强度。 |
| **UP_S** | Upper Shadow | $(High_t - \max(O,C)) / (High_t - Low_t)$ | **上影线率**：反映上方的抛压强度。 |
| **LO_S** | Lower Shadow | $(\min(O,C) - Low_t) / (High_t - Low_t)$ | **下影线率**：反映下方的支撑强度。 |
| **LOG_RET** | Log Return | $\ln(Close_t / Close_{t-1})$ | **对数收益率**：相比简单收益率，更符合正态分布假设，适合捕捉非线性特征。 |
| **SKEW** | Return Skewness | $RollingSkew(RET, 20)$ | **收益率偏度**：衡量分布的不对称性（左偏/右偏），捕捉尾部风险。 |
| **KURT** | Return Kurtosis | $RollingKurt(RET, 20)$ | **收益率峰度**：衡量分布的“肥尾”程度，捕捉极端行情发生的概率。 |
| **BB_W** | Bollinger Width | $(Upper - Lower) / Mid$ | **布林带宽度**：经典的波动率指标，反映行情的压缩（蓄势）与扩张（爆发）。 |
| **ATR** | Normalized ATR | $ATR_{14} / Close_t$ | **标准化ATR**：去量纲后的平均真实波幅，衡量价格波动的绝对强度。 |
| **V_SKEW** | Volatility Skew | $Corr(RET, \lvert RET \rvert, 20)$ | **波动率偏斜**：衡量“杠杆效应”。负值通常意味着下跌导致波动率放大（恐慌）。 |

> **注意**: 
> 1. 所有涉及除法的计算均包含 `1e-6` 极小值保护，防止除以零错误。
> 2. 高阶统计量（Skew, Kurt, Vol_Skew）通常基于过去 20 个交易日（约一个月）的窗口计算。
> **注意**: 所有涉及除法的计算均包含 `1e-6` 极小值保护，防止除以零错误。

-----

## 🧮 算子全集 (Operator Library)

本框架在 `operators.py` 中实现了以下 GPU 算子。所有算子均支持全矩阵并行计算。

### 1\. 基础数学算子 (Basic Math)

| 算子 | 描述 | Arity (参数个数) |
| :--- | :--- | :--- |
| `add(x, y)` | $x + y$ | 2 |
| `sub(x, y)` | $x - y$ | 2 |
| `mul(x, y)` | $x \times y$ | 2 |
| `protected_div(x, y)` | $x / y$ (如果 $y \approx 0$ 返回 1) | 2 |
| `abs_val(x)` | $\lvert x \rvert$ | 1 |
| `log_abs(x)` | $\ln(\lvert x \rvert + \epsilon)$ | 1 |
| `sqrt_abs(x)` | $\sqrt{\lvert x \rvert}$ | 1 |

### 2\. 时序算子 (Rolling Window Operators)

用于提取时间序列上的特征。窗口长度 $d$ 通常固定为 5, 10, 20。

| 算子 | 描述 | 逻辑 |
| :--- | :--- | :--- |
| `ts_mean(x, d)` | 滚动均值 | $\frac{1}{d} \sum_{i=0}^{d-1} x_{t-i}$ |
| `ts_std(x, d)` | 滚动标准差 | $\sqrt{Var(x_{t-d}...x_t)}$ |
| `ts_delta(x, d)` | 时序差分 | $x_t - x_{t-d}$ |
| `ts_max(x, d)` | 滚动最大值 | $\max(x_{t-d}...x_t)$ |
| `ts_min(x, d)` | 滚动最小值 | $\min(x_{t-d}...x_t)$ |
| `ts_rank(x, d)` | 时序排名 | $x_t$ 在过去 $d$ 天数据中的百分比排名 (0\~1) |
| `ts_corr(x, y, d)` | 滚动相关性 | $Corr(x_{t-d}...x_t, y_{t-d}...y_t)$ |
| `decay_linear(x, d)` | 线性衰减 | 加权平均，越近的数据权重越大 ($w_t=d, w_{t-1}=d-1...$) |

### 3\. 截面算子 (Cross-Sectional Operators)

用于在同一时间点比较不同股票的相对强弱。

| 算子 | 描述 | 逻辑 |
| :--- | :--- | :--- |
| `cs_rank(x)` | 截面排名 | 将当天的因子值转化为 0\~1 的百分比排名。 |
| `cs_scale(x)` | 截面 Z-Score | $(x - \mu_{date}) / \sigma_{date}$。去量纲，使不同分布的数据可以相加减。 |

-----

## ⚙️ 快速上手 (Quick Start)

### 1\. 环境安装

推荐使用 Conda 管理环境。

```bash
# 1. 创建环境
conda create -n gp_miner python=3.8
conda activate gp_miner

# 2. 安装 CUDA 支持的 CuPy (请先用 nvidia-smi 查看你的 CUDA 版本)
# 如果是 CUDA 12.x:
pip install cupy-cuda12x
# 如果是 CUDA 11.x:
pip install cupy-cuda11x

# 3. 安装其他依赖
pip install pandas numpy deap pyarrow
```

### 2\. 数据准备

请将你的数据放入 `data/` 目录：

  * `data/data.parquet`: 包含 `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`, `TurnOverValue` 等字段。
  * `data/ret_df.parquet`: 包含 `Date`, `Ticker`, `Target_Return`。

### 3\. 运行挖掘

```bash
python run.py
```

### 4\. 结果产出

程序运行结束后，结果会保存在 `GP/GPFactors/` 下自动生成的实验文件夹中。

  * **`formulas.csv`**: 挖掘出的因子公式、适应度得分。
  * **`All_Factors_Merged.parquet`**: 所有 Top 因子的宽表数据（已自动合并，可直接用于机器学习）。

-----

## 🔬 进阶：如何调参 (Tuning Guide)

修改 `config.py` 中的参数以控制挖掘方向：

  * **想要挖掘低频/基本面因子？**

      * 设置 `IC_METHOD = 'rank'`
      * 提高 `PENALTY_TURNOVER = 0.1` (严厉惩罚高换手)
      * 增大 `GENERATIONS = 30` (深度搜索)

  * **想要挖掘高频/量价因子？**

      * 设置 `IC_METHOD = 'rank'`
      * 降低 `PENALTY_TURNOVER = 0.01` (允许一定换手)
      * 重点关注 `ts_corr`, `ts_delta` 类算子

  * **发现因子全是 `CO_R` (过拟合)？**

      * 提高 `PENALTY_COMPLEXITY`
      * 或者在 `run.py` 中暂时屏蔽 `CO_R` 输入

-----

## ⚠️ 常见问题 (FAQ)

**Q: 为什么生成的因子全都是 -999 分？**
A: 通常是因为 `run.py` 中的 `pset` 参数个数与 `fitness.py` 中的输入个数不匹配。请检查 `pset = gp.PrimitiveSet("MAIN", 12)` 是否正确设置为 12。

**Q: 为什么我看不到显卡占用？**
A: GP 的计算是脉冲式的。只有在 Evaluation 阶段（每一代开始时）显卡才会满载，交叉变异阶段是 CPU 工作。可以使用 `watch -n 0.5 nvidia-smi` 观察。

**Q: 如何确认没有用到未来函数？**
A: 检查 `data_loader.py`。我们内置了 `target_shifted = target_raw.shift(-1)`，这确保了 Row(T) 的特征对应的是 Row(T+1) 的收益。

-----

## 🤝 贡献 (Contribution)

欢迎提交 Issue 或 Pull Request 来增加新的算子（如 Alpha191 中的复杂逻辑）或优化计算效率。
