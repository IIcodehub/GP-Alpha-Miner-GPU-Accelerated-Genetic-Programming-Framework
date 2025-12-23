
import cupy as cp
import numpy as np

def check_gpu_memory():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

# --- 基础算子 ---
def add(x, y): return cp.add(x, y)
def sub(x, y): return cp.subtract(x, y)
def mul(x, y): return cp.multiply(x, y)
def abs_val(x): return cp.abs(x)

def protected_div(left, right):
    # GPU 上避免除零崩溃，使用 where
    # 如果 right 近似 0，返回 1 (或者 0，视策略而定)
    return cp.where(cp.abs(right) < 1e-6, 1.0, left / right)

def log_abs(x):
    return cp.log(cp.abs(x) + 1e-6)

def sqrt_abs(x):
    return cp.sqrt(cp.abs(x))

# --- 时序算子 (Rolling) ---
# CuPy 没有直接的 rolling，使用 stride_tricks 或者卷积核加速
# 这是一个通用的 rolling_mean 实现
def ts_mean(x, window):
    # x shape: (Time, Stocks)
    # 使用卷积计算滑动平均
    kernel = cp.ones((window, 1), dtype=x.dtype) / window
    # mode='origin' 保持形状，但要注意前几行数据应该是无效的
    # 为了简单和速度，我们这里用切片循环（CuPy的循环比Numpy慢，但比CPU快）
    # 或者使用累计求和 (Cumsum) 技巧
    
    ret = cp.cumsum(x, axis=0)
    ret[window:] = ret[window:] - ret[:-window]
    result = ret / window
    
    # 前 window-1 行设为 0
    result[:window-1, :] = 0
    return result

def ts_delta(x, window):
    # x[t] - x[t-w]
    out = cp.full_like(x, 0)
    out[window:, :] = x[window:, :] - x[:-window, :]
    return out

def ts_std(x, window):
    # E[x^2] - (E[x])^2
    mean_x = ts_mean(x, window)
    mean_x2 = ts_mean(x**2, window)
    var = mean_x2 - mean_x**2

    return cp.sqrt(cp.maximum(var, 0))

# --- 截面算子 (Cross-Sectional) ---
def cs_rank(x):
    # axis=1
    # argsort两次得到排名 (0 ~ N-1)
    # 转换为 0~1
    raw_rank = cp.argsort(cp.argsort(x, axis=1), axis=1)
    return raw_rank / (x.shape[1] - 1)

def ts_max(x, window):
    # 利用 CuPy 的 sliding window view 比较耗显存，这里用简单的循环移位法（效率很高）
    # 逻辑：把过去 window 天的数据都平移出来，取 max
    ret = x.copy()
    for i in range(1, window):
        shifted = cp.roll(x, i, axis=0)
        # 还要把 roll 出来的头部脏数据设为极小值，防止干扰
        shifted[:i, :] = -cp.inf 
        ret = cp.maximum(ret, shifted)
    ret[:window-1, :] = 0 # 头部数据无效
    return ret

def ts_min(x, window):
    ret = x.copy()
    for i in range(1, window):
        shifted = cp.roll(x, i, axis=0)
        shifted[:i, :] = cp.inf 
        ret = cp.minimum(ret, shifted)
    ret[:window-1, :] = 0
    return ret

def ts_argmax(x, window):
    # 返回最大值距离现在的天数 (0 到 window-1)
    # 这是一个简化版实现，实盘中通常用 stride tricks
    max_val = ts_max(x, window)
    ret = cp.zeros_like(x)
    for i in range(window):
        shifted = cp.roll(x, i, axis=0)
        # 如果某天的值等于最大值，记录下 i
        # 注意：如果有多个最大值，这个逻辑会取最远的那个，或者最近的（取决于覆盖顺序）
        # 这里仅做演示，精确实现较复杂
        mask = (shifted == max_val)
        ret = cp.where(mask, i, ret)
    return ret

def ts_cov(x, y, window):
    # 滚动协方差: E[XY] - E[X]E[Y]
    mean_x = ts_mean(x, window)
    mean_y = ts_mean(y, window)
    mean_xy = ts_mean(x * y, window)
    return mean_xy - mean_x * mean_y

def ts_corr(x, y, window):
    # 滚动相关系数: Cov(X,Y) / (Std(X) * Std(Y))
    cov = ts_cov(x, y, window)
    std_x = ts_std(x, window)
    std_y = ts_std(y, window)
    
    # 保护除零
    return protected_div(cov, std_x * std_y)

def ts_rank(x, window):
    # 计算 x 在过去 window 天里的百分比排名
    # 暴力算法：当前值大于过去第 i 天的值，计数 +1
    count = cp.zeros_like(x)
    for i in range(window):
        shifted = cp.roll(x, i, axis=0)
        shifted[:i, :] = cp.nan # 处理边界
        # 如果 x > shifted, +1
        count += (x > shifted)
    
    return count / (window - 1)

def ts_decay_linear(x, window):
    # 加权平均：权重从 window 到 1
    # sum_weights = w*(w+1)/2
    weights = cp.arange(1, window + 1, dtype=cp.float32) # [1, 2, ..., w]
    sum_w = cp.sum(weights)
    
    res = cp.zeros_like(x)
    # 卷积逻辑：res[t] = x[t]*w + x[t-1]*(w-1) ...
    for i in range(window):
        shifted = cp.roll(x, i, axis=0)
        shifted[:i, :] = 0
        weight = (window - i) # 当前天权重最大
        res += shifted * weight
        
    return res / sum_w

def cs_scale(x):
    # 截面标准化 (Z-Score)
    mean = cp.mean(x, axis=1, keepdims=True)
    std = cp.std(x, axis=1, keepdims=True)
    return protected_div(x - mean, std)

def if_else(condition, true_val, false_val):
    """
    三元操作符：Where(condition > 0, true_val, false_val)
    """
    # 这里的 condition 通常是一个因子值，我们以 0 为界
    return cp.where(condition > 0, true_val, false_val)

def signed_power(x, exponent):
    """
    保持符号的幂运算。
    exponent 通常由 GP 生成常数，或者固定为 2, 3, 0.5
    """
    return cp.sign(x) * (cp.abs(x) ** exponent)

def step(x):
    """
    阶跃函数：x > 0 返回 1，否则 0
    用于构建二进制掩码 (Mask)
    """
    return cp.where(x > 0, 1.0, 0.0)

def ts_beta(x, y, window):
    # 计算 y 对 x 的回归斜率 beta
    # beta = Cov(x, y) / Var(x)
    cov = ts_cov(x, y, window)
    var_x = ts_std(x, window) ** 2
    return protected_div(cov, var_x)

def ts_resid(x, y, window):
    # 计算 y 对 x 回归的残差 (Residual)
    # Resid = y - beta * x - alpha (这里简化为去中心化的残差，即忽略常数项 alpha 的影响，或者认为均值回归)
    # 更严谨的写法：Resid = (y - mean_y) - beta * (x - mean_x)
    
    beta = ts_beta(x, y, window)
    
    mean_x = ts_mean(x, window)
    mean_y = ts_mean(y, window)
    
    # 纯特质波动
    return (y - mean_y) - beta * (x - mean_x)

def ts_argmax(x, window):
    # 返回窗口内最大值所在的索引距离现在的天数
    # 0 表示今天就是最大值，window-1 表示最大值在最远端
    # CuPy 没有直接的 rolling_argmax，需要用 stride_tricks 或者循环
    # 为了效率，这里使用循环扫描
    
    ret = cp.zeros_like(x)
    max_val = cp.full_like(x, -cp.inf)
    
    # 第一次扫描找最大值数值 (其实可以用 ts_max 代替，但为了同步索引需手动)
    # 但我们已经有 ts_max 了，可以直接利用
    rolling_max = ts_max(x, window)
    
    # 扫描过去 window 天，找到等于 max 的位置
    # 优先取最近的 (i从小到大)
    for i in range(window):
        shifted = cp.roll(x, i, axis=0)
        # 很多天可能都是最大值，我们记录最近的一次
        # 如果 shifted == rolling_max，且之前没找到过，则记录 i
        # 这里逻辑稍微复杂，简化为：如果是最大值，更新索引。
        # 为了取最近的，我们反向遍历或者利用 mask
        
        is_max = (shifted >= rolling_max - 1e-6) # 浮点容差
        # 如果 ret 还是 0 (默认)，且当前不是第0天(或者第0天也是max)，更新
        # 这是一个近似实现，只要捕捉到特征即可
        ret = cp.where(is_max, float(i), ret)
        
    return ret

def ts_argmin(x, window):
    # 同上，找最小值距离
    rolling_min = ts_min(x, window)
    ret = cp.zeros_like(x)
    for i in range(window):
        shifted = cp.roll(x, i, axis=0)
        is_min = (shifted <= rolling_min + 1e-6)
        ret = cp.where(is_min, float(i), ret)
    return ret

def ts_sum(x, window):
    # 也是利用 cumsum 实现 O(1)
    ret = cp.cumsum(x, axis=0)
    ret[window:] = ret[window:] - ret[:-window]
    ret[:window-1, :] = 0
    return ret

# --- 包装特定参数算子 ---
def ts_mean_5(x): return ts_mean(x, 5)
def ts_mean_20(x): return ts_mean(x, 20)
def ts_delta_1(x): return ts_delta(x, 1)
def ts_std_10(x): return ts_std(x, 10)
def ts_max_10(x): return ts_max(x, 10)
def ts_min_10(x): return ts_min(x, 10)
def ts_rank_10(x): return ts_rank(x, 10)
def ts_corr_10(x, y): return ts_corr(x, y, 10) 
def decay_10(x): return ts_decay_linear(x, 10)
def signed_sq(x): return signed_power(x, 2)
def signed_sqrt(x): return signed_power(x, 0.5)
def ts_beta_10(x, y): return ts_beta(x, y, 10)
def ts_resid_10(x, y): return ts_resid(x, y, 10)
def ts_argmax_10(x): return ts_argmax(x, 10)
def ts_argmin_10(x): return ts_argmin(x, 10)