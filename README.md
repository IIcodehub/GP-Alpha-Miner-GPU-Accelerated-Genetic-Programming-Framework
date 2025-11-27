

# 🧬 GP-Alpha-Miner: GPU-Accelerated Genetic Programming Framework

[](https://www.python.org/)
[](https://cupy.dev/)
[](https://github.com/DEAP/deap)
[](https://opensource.org/licenses/MIT)

**GP-Alpha-Miner** 是一个基于遗传规划（Genetic Programming, GP）的高性能量化因子挖掘框架。

该项目专为\*\*量化金融（Quantitative Finance）\*\*设计，利用 **DEAP** 进行进化算法调度，并深度结合 **CuPy** 实现全流程 **GPU 加速**。它能够在分钟级时间内处理全市场股票数据，自动挖掘出具有高夏普比率（ICIR）和低换手率的 Alpha 因子。

-----

## ✨ 核心特性 (Key Features)

  * 🚀 **GPU 极速计算**: 底层算子全部基于 `CuPy` 向量化实现，相比 CPU 单核计算快 50\~100 倍。
  * 📉 **严谨的数据对齐**: 内置 `shift(-1)` 逻辑，严格使用 T 日数据预测 T+1 日收益，杜绝未来函数（Look-ahead Bias）。
  * 🛡️ **多重防护机制**:
      * **去量纲化**: 引入 `cs_scale` (Z-Score) 和相对特征，防止机器依赖绝对价格。
      * **数值防护**: 三层 NaN/Inf 过滤网，防止梯度爆炸。
      * **反过拟合**: 支持换手率惩罚（Turnover Penalty）和复杂度惩罚（Parsimony Pressure）。
  * 📊 **丰富的算子库**: 内置时序（Time-Series）和截面（Cross-Sectional）算子，如 `ts_rank`, `ts_corr`, `cs_rank` 等。
  * 💾 **实验管理**: 自动根据参数生成实验文件夹，支持因子自动合并（Merge），便于后续机器学习模型使用。

-----

## 📂 项目结构 (Structure)

```text
GP-Alpha-Miner/
├── config.py           # 🎛️ [配置中心] 参数设置（种群大小、迭代次数、惩罚系数等）
├── data_loader.py      # 🏗️ [数据层] 数据清洗、特征工程、GPU显存搬运
├── operators.py        # 🧮 [算子库] 基于 CuPy 的 GPU 数学与金融算子实现
├── fitness.py          # ⚖️ [评估层] 适应度函数的核心逻辑 (IC/ICIR 计算)
├── run.py              # 🚀 [主程序] GP 进化循环入口，定义 DEAP 环境
├── utils.py            # 💾 [工具层] 结果保存、格式转换与因子合并
├── requirements.txt    # 📦 依赖列表
└── README.md           # 📖 说明文档
```

-----

## 🧠 核心逻辑详解 (How it Works)

### 1\. 数据流 (Data Pipeline)

1.  **加载 (Load)**: `data_loader.py` 读取 Parquet 格式的价量数据。
2.  **特征工程 (Feature Engineering)**: 在 CPU 端计算衍生特征（如 `RET`, `VWAP_D`, `HL_RATIO`），屏蔽原始绝对价格，防止伪因子。
3.  **上载 (Transfer)**: 将处理好的特征矩阵 ($T \times N$) 一次性搬运至 **GPU 显存**。
4.  **对齐 (Alignment)**: 目标收益率 `Target` 自动执行 `shift(-1)`，确保用于训练的是“未来的收益”。

### 2\. 进化循环 (Evolution Loop)

本项目遵循标准的遗传算法流程：

  * **初始化 (Initialization)**: 随机生成 2000 个公式树（个体）。
  * **评估 (Evaluation - GPU)**:
      * 将公式编译为 Python 函数。
      * 在 GPU 上进行大规模矩阵运算，得到因子值矩阵。
      * 计算 **Rank IC**（排序相关系数）和 **ICIR**。
      * 计算 **换手率惩罚**（基于自相关系数）。
      * 得出最终 `Fitness Score`。
  * **选择 (Selection)**: 使用锦标赛选择法（Tournament Selection），优胜劣汰。
  * **交叉与变异 (Crossover & Mutation)**: 交换子树或生成新节点，产生下一代。
  * **保存 (Save)**: 迭代结束后，将表现最好的因子从 GPU 拉回 CPU，保存为 Parquet 文件。

-----

## 🛠️ 安装指南 (Installation)

### 1\. 环境准备

建议使用 Conda 环境：

```bash
conda create -n gp_miner python=3.8
conda activate gp_miner
```

### 2\. 安装 CuPy (关键\!)

**注意**：请根据你的显卡 CUDA 版本安装对应的 CuPy 包，**不要直接 `pip install cupy`**（除非你配置好了完整的编译环境）。

检查 CUDA 版本：

```bash
nvidia-smi
```

  * 如果 CUDA 是 12.x: `pip install cupy-cuda12x`
  * 如果 CUDA 是 11.x: `pip install cupy-cuda11x`

### 3\. 安装其他依赖

```bash
pip install pandas numpy scipy deap pyarrow
```

-----

## 🚀 快速开始 (Quick Start)

### 1\. 准备数据

请将你的股票数据（Parquet格式）放入 `data/` 目录。

  * `data.parquet`: 包含 `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume` 等。
  * `ret_df.parquet`: 包含 `Date`, `Ticker`, `Return` (目标收益)。

### 2\. 配置参数

打开 `config.py` 修改核心参数：

```python
# 推荐实盘挖掘配置
POP_SIZE = 2000           # 种群大小
GENERATIONS = 20          # 迭代代数
IC_METHOD = 'rank'        # 使用 Rank IC
PENALTY_TURNOVER = 0.05   # 适度的换手率惩罚
OUTPUT_NUM = 20           # 保存前20个因子
```

### 3\. 运行挖掘

```bash
python run.py
```

### 4\. 查看结果

运行结束后，结果将保存在 `GP/GPFactors/` 下的带时间戳文件夹中：

  * `formulas.csv`: 所有挖掘出的因子公式列表。
  * `All_Factors_Merged.parquet`: 合并后的宽表数据，可直接用于机器学习训练。

-----

## 🧩 算子说明 (Operators)

系统内置了丰富的 Alpha 101 风格算子：

| 类型 | 算子名 | 描述 |
| :--- | :--- | :--- |
| **基础运算** | `add`, `sub`, `mul`, `protected_div` | 加减乘除 (带除零保护) |
| **数学函数** | `log_abs`, `sqrt_abs`, `abs_val` | 对数、开方、绝对值 |
| **时序算子** | `ts_mean(x, d)` | 过去 d 天的均值 |
| | `ts_delta(x, d)` | 相比 d 天前的变化量 |
| | `ts_rank(x, d)` | 过去 d 天的百分比排名 (**核心算子**) |
| | `ts_corr(x, y, d)` | 过去 d 天两个序列的相关系数 |
| | `ts_max(x, d)` / `ts_min` | 过去 d 天的最值 |
| | `decay_linear(x, d)` | 线性衰减加权平均 |
| **截面算子** | `cs_rank(x)` | 每日截面排序 (0\~1) |
| | `cs_scale(x)` | 每日截面 Z-Score 标准化 (**去量纲神器**) |

-----

## ⚠️ 注意事项 (Notes)

1.  **显存占用**: `POP_SIZE` 越大，瞬间显存占用越高。如果遇到 OOM (Out Of Memory) 错误，请减小种群大小或减少特征数量。
2.  **数据泄露**: 挖掘出的因子 IC 如果超过 0.15，请务必检查 `data_loader.py` 中的 `shift(-1)` 逻辑是否生效。
3.  **单进程**: 由于 CUDA 上下文限制，本项目**不支持** `multiprocessing` 多进程并行，请直接运行脚本。

-----

## 🤝 贡献与反馈

如果你是量化初学者，欢迎 Fork 本项目进行修改实验！
如果是 Bug 反馈或功能建议，请提交 Issue。

**Happy Alpha Mining\! 💰**