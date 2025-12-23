import pandas as pd
import numpy as np
import cupy as cp
import config

class DataPortalGPU:
    def __init__(self):
        self.load_and_process()

    def load_and_process(self):
        print("[Data] Loading Parquet (CPU)...")
        df_raw = pd.read_parquet(config.RAW_DATA_PATH)
        ret_df = pd.read_parquet(config.RET_DATA_PATH)

        # 1. ID 清洗
        df_raw['Ticker'] = df_raw['Ticker'].astype(str).str[:6]
        ret_df['SecuCode'] = ret_df['SecuCode'].astype(str).str.zfill(6)
        
        df_raw = df_raw.rename(columns={'Date': 'date', 'Ticker': 'asset'})
        ret_df = ret_df.rename(columns={'TradingDay': 'date', 'SecuCode': 'asset'})
        
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        ret_df['date'] = pd.to_datetime(ret_df['date'])

        # 2. 对齐索引
        common_dates = sorted(list(set(df_raw['date']) & set(ret_df['date'])))
        
        pivot_temp = df_raw[df_raw['date'].isin(common_dates)] \
            .pivot(index='date', columns='asset', values='ClosePrice') \
            .reindex(common_dates)
        
        self.dates = pivot_temp.index
        self.assets = pivot_temp.columns
        
        # 3. 特征工程 (CPU)
        print("[Data] Feature Engineering (Adding Non-linear & Volatility Features)...")
        
        def get_df_matrix(col_name):
            # 获取 DataFrame 格式以便使用 rolling 函数
            return df_raw[df_raw['date'].isin(common_dates)] \
                .pivot(index='date', columns='asset', values=col_name) \
                .reindex(common_dates).reindex(columns=self.assets) \
                .ffill().fillna(0)

        # 加载原始 DataFrame
        df_open  = get_df_matrix('OpenPrice')
        df_high  = get_df_matrix('HighPrice')
        df_low   = get_df_matrix('LowPrice')
        df_close = get_df_matrix('ClosePrice')
        df_vol   = get_df_matrix('TurnOverVolume')
        df_amt   = get_df_matrix('TurnOverValue')
        
        # 转为 numpy 用于基础计算
        raw_close = df_close.values.astype(np.float32)
        raw_rate = get_df_matrix('TurnOverRate').values.astype(np.float32)
        raw_cap  = get_df_matrix('FloatMarketValue').values.astype(np.float32)

        # --- A. 基础相对特征 ---
        ret = np.zeros_like(raw_close)
        ret[1:] = raw_close[1:] / (raw_close[:-1] + 1e-6) - 1.0
        
        open_gap = np.zeros_like(raw_close)
        open_gap[1:] = df_open.values[1:] / (raw_close[:-1] + 1e-6) - 1.0
        
        hl_ratio = df_high.values / (df_low.values + 1e-6) - 1.0
        co_ratio = raw_close / (df_open.values + 1e-6) - 1.0
        
        log_vol = np.log(df_vol.values + 1.0)
        log_cap = np.log(raw_cap + 1.0)
        
        # VWAP & AMIHUD
        vwap = df_amt.values / (df_vol.values + 1.0)
        vwap_dist = raw_close / (vwap + 1e-6) - 1.0
        amihud = np.abs(ret) / (df_amt.values + 1e6) * 1e9
        
        # K线结构
        day_range = df_high.values - df_low.values + 1e-6
        body_r = np.abs(raw_close - df_open.values) / day_range
        up_shd = (df_high.values - np.maximum(df_open.values, raw_close)) / day_range
        lo_shd = (np.minimum(df_open.values, raw_close) - df_low.values) / day_range

        # --- B. 新增高级特征 (Non-linear & Volatility) ---
        # 窗口设为 20 (一个月)
        W = 20
        
        # 1. LOG_RET: 对数收益率 (非线性基础)
        # ln(Pt / Pt-1)
        log_ret = np.zeros_like(raw_close)
        log_ret[1:] = np.log(raw_close[1:] / (raw_close[:-1] + 1e-6))

        # 2. SKEW: 收益率偏度 (20日)
        # 使用 Pandas Rolling 加速
        df_ret = pd.DataFrame(ret, index=self.dates, columns=self.assets)
        skew = df_ret.rolling(W).skew().fillna(0).values.astype(np.float32)

        # 3. KURT: 收益率峰度 (20日)
        kurt = df_ret.rolling(W).kurt().fillna(0).values.astype(np.float32)

        # 4. BB_WIDTH: 布林带宽度
        # (Upper - Lower) / Mid
        # Upper = Mean + 2*Std
        rolling_mean = df_close.rolling(W).mean()
        rolling_std = df_close.rolling(W).std()
        # Width = (4 * Std) / Mean
        bb_width = (4 * rolling_std / (rolling_mean + 1e-6)).fillna(0).values.astype(np.float32)

        # 5. ATR: 平均真实波幅 (14日)
        # TR = Max(H-L, |H-Cp|, |L-Cp|)
        prev_close = df_close.shift(1).fillna(method='bfill')
        tr1 = df_high - df_low
        tr2 = (df_high - prev_close).abs()
        tr3 = (df_low - prev_close).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(14).mean().fillna(0).values.astype(np.float32)
        # 为了去量纲，通常建议用 ATR / Close 标准化，这里保留原始 ATR 或 ATR/Close
        # 建议：标准化 ATR (NATR)
        atr_norm = atr / (raw_close + 1e-6)

        # 6. VOL_SKEW: 波动率偏斜 (Upside Vol vs Downside Vol)
        # 简单定义：下行波动率 / 上行波动率 (反映恐慌程度)
        # 或者：Ret 和 |Ret| 的相关性
        # 这里实现：(下行标准差 - 上行标准差) / 总标准差
        # 向量化实现 Volatility Skew (Corr(Ret, |Ret|))
        # 如果相关性为负，说明下跌导致波动放大（常见的波动率偏斜）
        df_abs_ret = df_ret.abs()
        vol_skew = df_ret.rolling(W).corr(df_abs_ret).fillna(0).values.astype(np.float32)

        # 4. 上传至 GPU
        # 总共 18 个特征
        print("[Data] Moving 18 features to GPU Memory...")
        self.features = {}
        # 原有
        self.features['RET']      = cp.asarray(ret)
        self.features['OPEN_GAP'] = cp.asarray(open_gap)
        self.features['HL_RATIO'] = cp.asarray(hl_ratio)
        self.features['CO_RATIO'] = cp.asarray(co_ratio)
        self.features['LOG_VOL']  = cp.asarray(log_vol)
        self.features['TO_RATE']  = cp.asarray(raw_rate)
        self.features['LOG_CAP']  = cp.asarray(log_cap)
        self.features['VWAP_D']   = cp.asarray(vwap_dist)
        self.features['AMIHUD']   = cp.asarray(amihud)
        self.features['BODY_R']   = cp.asarray(body_r)
        self.features['UP_SHD']   = cp.asarray(up_shd)
        self.features['LO_SHD']   = cp.asarray(lo_shd)
        self.features['LOG_RET']  = cp.asarray(log_ret)
        self.features['SKEW']     = cp.asarray(skew)
        self.features['KURT']     = cp.asarray(kurt)
        self.features['BB_WIDTH'] = cp.asarray(bb_width)
        self.features['ATR']      = cp.asarray(atr_norm) 
        self.features['VOL_SKEW'] = cp.asarray(vol_skew)

        # Target 处理 (Shift -1 防未来)
        print("[Data] aligning Target (Shift -1)...")
        target_col = 'ret_open5twap'
        target_raw = ret_df[ret_df['date'].isin(common_dates)] \
            .pivot(index='date', columns='asset', values=target_col) \
            .reindex(common_dates).reindex(columns=self.assets)
        
        target_shifted = target_raw.shift(-1).fillna(0)
        target_cpu = target_shifted.values.astype(np.float32)
        self.target = cp.asarray(target_cpu)
        
        print(f"[Data] Done. Matrix Shape: {self.target.shape}")