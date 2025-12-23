
import cupy as cp
import numpy as np
import config

def calculate_fitness_gpu(individual, data_portal, toolbox):
    """
    计算适应度函数 (GPU版)
    包含：因子计算 -> 数值清洗 -> 换手率估算 -> IC计算 -> 综合打分
    """
    
    # =========================================================================
    # Step 1: 编译公式并计算因子值 (Execution)
    # =========================================================================
    try:
        func = toolbox.compile(expr=individual)
        # [修改点] 这里传入的参数必须和 Data_Loader 的 key 以及 Run.py 的 ARG 顺序一致
        factor = func(
            data_portal.features['RET'],        # ARG0
            data_portal.features['OPEN_GAP'],   # ARG1
            data_portal.features['HL_RATIO'],   # ARG2
            data_portal.features['CO_RATIO'],   # ARG3
            data_portal.features['LOG_VOL'],    # ARG4
            data_portal.features['TO_RATE'],    # ARG5
            data_portal.features['LOG_CAP'],    # ARG6
            data_portal.features['VWAP_D'],     # ARG7
            data_portal.features['AMIHUD'],     # ARG8
            data_portal.features['BODY_R'],     # ARG9
            data_portal.features['UP_SHD'],     # ARG10
            data_portal.features['LO_SHD'],      # ARG11
            data_portal.features['LOG_RET'],    # ARG12
            data_portal.features['SKEW'],       # ARG13
            data_portal.features['KURT'],       # ARG14
            data_portal.features['BB_WIDTH'],   # ARG15
            data_portal.features['ATR'],        # ARG16
            data_portal.features['VOL_SKEW']    # ARG17
        )
    except Exception:
        return -999.0,

    # =========================================================================
    # Step 2: 数值清洗与防护 (Sanity Check & Clipping) - 【关键修复 NAN】
    # =========================================================================
    # 1. 处理 NaN 和 Inf
    # 将 nan 变 0，将 inf 变极大值
    factor = cp.nan_to_num(factor, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 2. 强力截断 (Clipping)
    # GP 产生的公式极其狂野（例如 Volume^5），容易超出 float32 上限。
    # 我们将因子值限制在 [-1e6, 1e6] 之间。
    # 对于 Rank IC，这不影响排序；对于 Normal IC，这能防止方差爆炸。
    factor = cp.clip(factor, -1e6, 1e6)
    
    # 3. 零方差过滤 (Constant Check)
    # 如果因子是常数 (Std < 1e-5)，无预测能力，直接淘汰
    if cp.std(factor) < 1e-5:
        return -999.0,

    # =========================================================================
    # Step 3: 计算换手率代理 (Turnover Proxy via Auto-Correlation)
    # =========================================================================
    # 逻辑：计算因子 t 时刻与 t+1 时刻的余弦相似度
    f_curr = factor[:-1]
    f_next = factor[1:]
    
    # 去均值
    f_curr_demean = f_curr - cp.mean(f_curr, axis=1, keepdims=True)
    f_next_demean = f_next - cp.mean(f_next, axis=1, keepdims=True)
    
    # Cosine Similarity
    num = cp.sum(f_curr_demean * f_next_demean, axis=1)
    den = cp.sqrt(cp.sum(f_curr_demean**2, axis=1) * cp.sum(f_next_demean**2, axis=1))
    
    # 防止分母为 0
    daily_autocorr = num / (den + 1e-9)
    
    # 处理可能的 nan (如果某天方差为0)
    daily_autocorr = cp.nan_to_num(daily_autocorr, nan=0.0)
    avg_autocorr = cp.mean(daily_autocorr)
    
    # 惩罚项计算：自相关越接近 1 (稳定)，惩罚越小
    turnover_penalty = (1.0 - float(avg_autocorr)) * config.PENALTY_TURNOVER

    # =========================================================================
    # Step 4: 计算 IC (Normal / Rank)
    # =========================================================================
    target = data_portal.target

    if config.IC_METHOD == 'rank':
        # === Rank IC (Spearman) ===
        # 原理: SpearmanCorr = PearsonCorr(Rank(X), Rank(Y))
        # axis=1 表示对每天的截面做排序
        input_x = cp.argsort(cp.argsort(factor, axis=1), axis=1)
        input_y = cp.argsort(cp.argsort(target, axis=1), axis=1)
        
    elif config.IC_METHOD == 'normal':
        # === Normal IC (Pearson) ===
        # 使用原始值，但做 3倍标准差去极值，保护 Pearson 不被 Outliers 扭曲
        mean_f = cp.mean(factor, axis=1, keepdims=True)
        std_f = cp.std(factor, axis=1, keepdims=True)
        input_x = cp.clip(factor, mean_f - 4*std_f, mean_f + 4*std_f)
        input_y = target
        
    # === 通用 Pearson 相关系数计算逻辑 ===
    # 1. 中心化
    x_demean = input_x - cp.mean(input_x, axis=1, keepdims=True)
    y_demean = input_y - cp.mean(input_y, axis=1, keepdims=True)
    
    # 2. 协方差 (分子)
    cov = cp.sum(x_demean * y_demean, axis=1)
    
    # 3. 标准差乘积 (分母)
    var_x = cp.sum(x_demean**2, axis=1)
    var_y = cp.sum(y_demean**2, axis=1)
    
    # 4. 每日 IC 序列
    daily_ic = cov / (cp.sqrt(var_x * var_y) + 1e-9)
    
    # [防护] 再次清洗 daily_ic，防止因为输入全0导致的 nan
    daily_ic = cp.nan_to_num(daily_ic, nan=0.0)
    
    # =========================================================================
    # Step 5: 综合评分 (Scoring)
    # =========================================================================
    ic_mean = cp.mean(daily_ic)
    ic_std = cp.std(daily_ic)
    
    icir = 0.0
    # 防止除以极小值
    if ic_std > 1e-7:
        icir = ic_mean / ic_std
        
    # 处理反向因子：如果 IC 均值为负
    # 策略 1: 惩罚它 (我们只想要正向因子) -> 当前采用
    # 策略 2: 取绝对值 (我们接受反向因子) -> raw_score = abs(icir)
    raw_score = float(icir)
    if ic_mean < 0:
        raw_score = raw_score * 0.1 # 给予重罚

    # 复杂度惩罚 
    complexity_penalty = len(individual) * config.PENALTY_COMPLEXITY
    score = raw_score - turnover_penalty - complexity_penalty
    
    # =========================================================================
    # Step 6: 最终兜底 (Final Safety Net)
    # =========================================================================
    # 这一步至关重要：如果 float 转换或者计算过程产生了 nan/inf 必须拦截，否则 DEAP 的 avg/max 统计会全部变成 nan
    if np.isnan(score) or np.isinf(score):
        return -999.0,
    
    return score,