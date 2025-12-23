import os
import pandas as pd
import cupy as cp
import datetime
import config

def merge_output_files(save_dir):

    print(f"\n[Output] Merging individual factor files in {save_dir}...")
    
    # 1. 找到所有因子文件
    files = [f for f in os.listdir(save_dir) if f.startswith('AlphaFactor') and f.endswith('.parquet')]
    
    if not files:
        print("  No factor files found to merge.")
        return

    try:
        files.sort(key=lambda x: int(x.replace('AlphaFactor', '').replace('.parquet', '')))
    except:
        files.sort()

    dfs = []
    print(f"  Found {len(files)} files. Reading and combining...")
    
    # 3. 读取并拼接
    for f in files:
        path = os.path.join(save_dir, f)
        # 读取单个因子文件 (TradingDay, SecuCode, AlphaFactorX)
        df = pd.read_parquet(path)
        
        # 将 [TradingDay, SecuCode] 设为索引，以便横向 Concat
        df = df.set_index(['TradingDay', 'SecuCode'])
        dfs.append(df)
    
    if dfs:
        # 4. 横向合并 (axis=1)
        merged_df = pd.concat(dfs, axis=1)
        
        # 重置索引，变回普通列
        merged_df = merged_df.reset_index()
        
        # 5. 保存合并后的文件
        parquet_path = os.path.join(save_dir, "All_Factors_Merged.parquet")
        merged_df.to_parquet(parquet_path, index=False)
        
        
        print(f"[Success] Merged data saved to:\n    -> {parquet_path}")

def save_best_factors(hof, data_portal, toolbox):
    """
    保存名人堂(Hall of Fame)中的因子到特定参数的子文件夹下。
    """
    # 1. 构建基于参数的子文件夹名称
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir_name = (
        f"Pop{config.POP_SIZE}_Gen{config.GENERATIONS}_"
        f"Turn{config.PENALTY_TURNOVER}_Cplx{config.PENALTY_COMPLEXITY}_"
        f"{config.IC_METHOD}_{timestamp}"
    )
    
    # 拼接完整路径
    save_dir = os.path.join(config.OUTPUT_DIR, sub_dir_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"[Output] Created experiment directory: {save_dir}")

    records = []
    
    # 2. 确定要保存的因子数量
    num_to_save = min(len(hof), config.OUTPUT_NUM)
    print(f"[Output] Saving top {num_to_save} factors...")
    
    dates = data_portal.dates
    assets = data_portal.assets
    
    # 只遍历前 num_to_save 个因子
    for i in range(num_to_save):
        ind = hof[i]
        name = f"AlphaFactor{i+1}"
        
        try:
            # --- GPU 计算 ---
            func = toolbox.compile(expr=ind)
            # 确保参数列表与 fitness.py 和 run.py 一致
            factor_gpu = func(
                data_portal.features['RET'],
                data_portal.features['OPEN_GAP'],
                data_portal.features['HL_RATIO'],
                data_portal.features['CO_RATIO'],
                data_portal.features['LOG_VOL'],
                data_portal.features['TO_RATE'],
                data_portal.features['LOG_CAP'],
                data_portal.features['VWAP_D'],
                data_portal.features['AMIHUD'],
                data_portal.features['BODY_R'],
                data_portal.features['UP_SHD'],
                data_portal.features['LO_SHD'],
                data_portal.features['LOG_RET'],
                data_portal.features['SKEW'],
                data_portal.features['KURT'],
                data_portal.features['BB_WIDTH'],
                data_portal.features['ATR'],
                data_portal.features['VOL_SKEW']
            )
            
            # --- 数据搬运: GPU -> CPU ---
            factor_cpu = cp.asnumpy(factor_gpu)
            
            # --- 格式化保存 ---
            df_wide = pd.DataFrame(factor_cpu, index=dates, columns=assets)
            
            # 堆叠为长表 [TradingDay, SecuCode, Value]
            df_long = df_wide.stack().reset_index()
            df_long.columns = ['TradingDay', 'SecuCode', name]
            
            # 保存 Parquet 到子文件夹
            save_path = os.path.join(save_dir, f"{name}.parquet")
            df_long.to_parquet(save_path, index=False)
            
            # 记录元数据
            records.append({
                'Rank': i + 1,
                'Name': name,
                'Formula': str(ind),
                'Fitness': float(ind.fitness.values[0])
            })
            print(f"  -> Saved {name}")
            
        except Exception as e:
            print(f"  [Error] Failed to save {name}: {e}")
            
    # 3. 保存公式汇总表
    csv_path = os.path.join(save_dir, "formulas.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"[Output] Formulas saved.")
    
    merge_output_files(save_dir)

    print(f"[Output] Experiment completed: {save_dir}")