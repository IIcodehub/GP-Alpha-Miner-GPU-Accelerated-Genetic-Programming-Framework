
# ----------------- 路径配置 -----------------
# 原始行情数据（Parquet格式）
RAW_DATA_PATH = 'data/data.parquet'
# 目标收益数据（Parquet格式，用于计算IC）
RET_DATA_PATH = 'data/ret_df.parquet'
# 结果保存目录
OUTPUT_DIR = 'GP/GPFactorsRound3'

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
PENALTY_TURNOVER = 0.01

# 复杂度惩罚系数：值越大，越倾向于挖掘短公式。
# 防止 "Bloat"（公式膨胀）现象。
PENALTY_COMPLEXITY = 0.02

# ----------------- 硬件开关 -----------------
# 是否使用 GPU 加速。如果设为 False，代码会报错（因为 operators 写死了 cupy）
USE_GPU = True

OUTPUT_NUM = 20

# ----------------- 热启动配置 (Warm Start) -----------------
# 是否开启热启动模式
# True: 种群初始化时使用 seeds.py 中的公式
# False: 使用传统的 genHalfAndHalf 随机生成
USE_WARM_START = True

# 结构约束模式
# True: 变异时仅改变叶子节点（特征）或参数，不改变公式结构（论文中的 Point Mutation）
# False: 允许传统的子树替换（可能会破坏好结构）
RESTRICT_STRUCTURE = False

# ----------------- 结构约束专用参数 (Internal Probabilities) -----------------
# 在同构交叉中，两个相同位置的叶子节点发生交换的概率
# 0.5 意味着平均有一半的叶子会被交换 (类似 Uniform Crossover)
P_LEAF_SWAP = 0.5

# 在点突变中，每个叶子节点发生突变的概率
# 0.2 意味着一个公式里平均有 20% 的特征会被替换
P_LEAF_MUTATION = 0.2