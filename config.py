
# ----------------- 路径配置 -----------------
# 原始行情数据（Parquet格式）
RAW_DATA_PATH = 'data/data.parquet'
# 目标收益数据（Parquet格式，用于计算IC）
RET_DATA_PATH = 'data/ret_df.parquet'
# 结果保存目录
OUTPUT_DIR = 'GP/GPFactors'

# ----------------- 进化算法参数 (Hyperparameters) -----------------
# 种群大小：每一代有多少个公式存活。
# 建议：测试用 100，实盘挖掘建议 500~2000。种群越大，多样性越好，但速度越慢。
POP_SIZE = 2000

# 迭代代数：进化多少轮。
# 建议：测试用 5~20，实盘建议 20~50。通常 20代以后提升就不明显了。
GENERATIONS = 20

# 锦标赛规模：每次选秀从多少个个体里挑最好的。
# 越大，选择压力越大（收敛快，容易陷入局部最优）；越小，越随机（多样性好，收敛慢）。
TOURNAMENT_SIZE = 3

# ----------------- 适应度评估参数 (Fitness Logic) -----------------
# IC计算模式：
# 'rank': Spearman排序相关系数。对异常值不敏感，最推荐。
# 'normal': Pearson线性相关系数。计算极快，但会被离群值干扰。
IC_METHOD = 'rank'

# 换手率惩罚系数：值越大，越倾向于挖掘低换手因子。
# 如果挖出的因子全是 Close/Open 这种高频噪音，调大它。
PENALTY_TURNOVER = 0.05

# 复杂度惩罚系数：值越大，越倾向于挖掘短公式。
# 防止 "Bloat"（公式膨胀）现象。
PENALTY_COMPLEXITY = 0.05

# ----------------- 硬件开关 -----------------
# 是否使用 GPU 加速。如果设为 False，代码会报错（因为 operators 写死了 cupy）
USE_GPU = True

OUTPUT_NUM = 20