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
        
        # 保存元数据以便后续还原
        pivot_temp = df_raw[df_raw['date'].isin(common_dates)] \
            .pivot(index='date', columns='asset', values='ClosePrice') \
            .reindex(common_dates)
        
        self.dates = pivot_temp.index
        self.assets = pivot_temp.columns
        
        # 3. 特征工程：计算相对指标 (Shielding Raw Prices)
        print("[Data] Feature Engineering on CPU (Shielding Raw Prices)...")
        
        # 辅助函数：快速获取 Numpy 矩阵
        def get_numpy_matrix(col_name):
            mat = df_raw[df_raw['date'].isin(common_dates)] \
                .pivot(index='date', columns='asset', values=col_name) \
                .reindex(common_dates).reindex(columns=self.assets) \
                .ffill().fillna(0).values.astype(np.float32)
            return mat

        # 加载原始数据
        raw_open = get_numpy_matrix('OpenPrice')
        raw_high = get_numpy_matrix('HighPrice')
        raw_low  = get_numpy_matrix('LowPrice')
        raw_close = get_numpy_matrix('ClosePrice')
        raw_vol  = get_numpy_matrix('TurnOverVolume')
        raw_rate = get_numpy_matrix('TurnOverRate')
        raw_cap  = get_numpy_matrix('FloatMarketValue')
        raw_amt  = get_numpy_matrix('TurnOverValue')
        # --- 计算衍生特征 (避免使用绝对价格) ---
        
        # [A] RET: 日收益率 (Close / PrevClose - 1)
        # 错位计算，处理除0
        ret = np.zeros_like(raw_close)
        ret[1:] = raw_close[1:] / (raw_close[:-1] + 1e-6) - 1.0
        
        # [B] OPEN_GAP: 开盘跳空 (Open / PrevClose - 1)
        open_gap = np.zeros_like(raw_open)
        open_gap[1:] = raw_open[1:] / (raw_close[:-1] + 1e-6) - 1.0
        
        # [C] HL_RATIO: 日内振幅 (High / Low - 1)
        hl_ratio = raw_high / (raw_low + 1e-6) - 1.0
        
        # [D] CO_RATIO: 日内涨跌 (Close / Open - 1)
        co_ratio = raw_close / (raw_open + 1e-6) - 1.0
        
        # [E] LOG_VOL: 对数成交量 (压缩量级)
        log_vol = np.log(raw_vol + 1.0)
        
        # [F] LOG_CAP: 对数市值
        log_cap = np.log(raw_cap + 1.0)
        
        # [G] TO_RATE: 换手率 (本身就是比率，保留)
        to_rate = raw_rate
        # 1. VWAP_DIST: 收盘价相对于均价的偏离度
        # VWAP = Amount / Volume
        vwap = raw_amt / (raw_vol + 1.0)
        vwap_dist = raw_close / (vwap + 1e-6) - 1.0
        
        # 2. AMIHUD: 非流动性指标 (|Ret| / Amount)
        # 含义：单位成交额带来的涨跌幅。值越大越缺乏流动性。
        # 乘以 1e9 是为了让数值量级正常化
        amihud = np.abs(ret) / (raw_amt + 1e6) * 1e9
        
        # 3. K线结构
        day_range = raw_high - raw_low + 1e-6
        
        # BODY_R: 实体率 (Abs(Close-Open) / Range)
        body_r = np.abs(raw_close - raw_open) / day_range
        
        # UP_SHD: 上影线率 ((High - Max(O,C)) / Range)
        up_shd = (raw_high - np.maximum(raw_open, raw_close)) / day_range
        
        # LO_SHD: 下影线率 ((Min(O,C) - Low) / Range)
        lo_shd = (np.minimum(raw_open, raw_close) - raw_low) / day_range

        # 4. 上传至 GPU
        print("[Data] Moving features to GPU Memory...")
        self.features = {}
        self.features['RET']      = cp.asarray(ret)
        self.features['OPEN_GAP'] = cp.asarray(open_gap)
        self.features['HL_RATIO'] = cp.asarray(hl_ratio)
        self.features['CO_RATIO'] = cp.asarray(co_ratio)
        self.features['LOG_VOL']  = cp.asarray(log_vol)
        self.features['TO_RATE']  = cp.asarray(to_rate)
        self.features['LOG_CAP']  = cp.asarray(log_cap)
        self.features['VWAP_D']   = cp.asarray(vwap_dist)
        self.features['AMIHUD']   = cp.asarray(amihud)
        self.features['BODY_R']   = cp.asarray(body_r)
        self.features['UP_SHD']   = cp.asarray(up_shd)
        self.features['LO_SHD']   = cp.asarray(lo_shd)
        # 处理 Target
        target_col = 'ret_open5twap'
        target_cpu = ret_df[ret_df['date'].isin(common_dates)] \
            .pivot(index='date', columns='asset', values=target_col) \
            .reindex(common_dates).reindex(columns=self.assets) \
            .fillna(0).values.astype(np.float32)
        
        self.target = cp.asarray(target_cpu)
        
        print(f"[Data] Done. Matrix Shape: {self.target.shape}")