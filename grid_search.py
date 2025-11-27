import itertools
import config
from run import main as run_main
import time

# ==============================================================================
# 1. 定义参数网格 (Parameter Grid)
# 在这里列出你想尝试的所有参数值，脚本会自动组合它们
# ==============================================================================
param_grid = {
    'POP_SIZE': [2000],
    
    'GENERATIONS': [20],
    
    'PENALTY_TURNOVER': [0.01, 0.05, 0.1],
    
    'PENALTY_COMPLEXITY': [0.001, 0.002, 0.005],
    
    'IC_METHOD': ['rank'] 
}

def run_grid_search():

    keys = param_grid.keys()
    values = param_grid.values()

    combinations = list(itertools.product(*values))
    total_experiments = len(combinations)
    
    print(f"===========================================================")
    print(f">>> Grid Search Started. Total Configurations: {total_experiments}")
    print(f"===========================================================\n")
    
    start_time_global = time.time()

    for i, combination in enumerate(combinations):
        # 将 tuple 转回 dict，方便查看
        current_params = dict(zip(keys, combination))
        
        print(f"--- Running Experiment {i+1}/{total_experiments} ---")
        print(f"Parameters: {current_params}")
        
        # 3. 动态修改 config 里的变量
        # 使用 setattr 动态赋值，不用手动写 config.X = Y
        for key, value in current_params.items():
            setattr(config, key, value)
            
        # 4. 执行主程序
        try:
            start_time = time.time()
            run_main()
            elapsed = time.time() - start_time
            print(f">>> Experiment {i+1} Finished in {elapsed:.1f}s\n")
            
        except Exception as e:
            print(f">>> Experiment {i+1} Failed! Error: {e}\n")
            # 捕获异常，防止一组报错导致整个网格搜索停止
            continue

    total_time = (time.time() - start_time_global) / 3600
    print(f"===========================================================")
    print(f"All Experiments Completed in {total_time:.2f} hours.")
    print(f"Please check {config.OUTPUT_DIR} for results.")

if __name__ == "__main__":
    run_grid_search()