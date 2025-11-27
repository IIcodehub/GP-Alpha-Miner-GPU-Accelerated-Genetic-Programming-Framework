
import random
import operator
import numpy as np
from functools import partial
from deap import base, creator, tools, gp, algorithms
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
import config
import operators
import data_loader
import fitness
import utils

def main():
    # 1. 加载数据 (自动进入 GPU)
    # 请确保 config.py 里的路径是正确的
    data = data_loader.DataPortalGPU()
    
    # 2. GP 环境设置
    pset = gp.PrimitiveSet("MAIN", 12)

    pset.renameArguments(
        ARG0='RET',       # 收益率
        ARG1='GAP',       # 跳空
        ARG2='HL_R',      # 振幅
        ARG3='CO_R',      # 日内涨幅
        ARG4='L_VOL',     # 对数成交量
        ARG5='TO_RATE',   # 换手率
        ARG6='L_CAP',     # 对数市值
        ARG7='VWAP_D',    # 均价偏离
        ARG8='AMI',       # Amihud
        ARG9='BODY',      # 实体率
        ARG10='UP_S',     # 上影线
        ARG11='LO_S'      # 下影线
    )
    # 注册 GPU 算子
    pset.addPrimitive(operators.add, 2)
    pset.addPrimitive(operators.sub, 2)
    pset.addPrimitive(operators.mul, 2)
    pset.addPrimitive(operators.protected_div, 2)
    pset.addPrimitive(operators.abs_val, 1)
    pset.addPrimitive(operators.log_abs, 1)
    pset.addPrimitive(operators.sqrt_abs, 1)
    pset.addPrimitive(operators.ts_mean_5, 1)
    pset.addPrimitive(operators.ts_mean_20, 1)
    pset.addPrimitive(operators.ts_delta_1, 1)
    pset.addPrimitive(operators.ts_std_10, 1)
    pset.addPrimitive(operators.cs_rank, 1)
    pset.addPrimitive(operators.ts_max_10, 1)
    pset.addPrimitive(operators.ts_min_10, 1)
    pset.addPrimitive(operators.ts_rank_10, 1)
    pset.addPrimitive(operators.decay_10, 1)
    pset.addPrimitive(operators.cs_scale, 1)


    pset.addPrimitive(operators.ts_corr_10, 2)
    pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

    # Creator
    if hasattr(creator, 'FitnessMax'): del creator.FitnessMax
    if hasattr(creator, 'Individual'): del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    # 挂载 GPU 评估函数
    toolbox.register("evaluate", partial(fitness.calculate_fitness_gpu, data_portal=data, toolbox=toolbox))
    
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # 限制树深度
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    # 3. 运行
    pop = toolbox.population(n=config.POP_SIZE)
    hof_size = max(config.OUTPUT_NUM, 50) 
    hof = tools.HallOfFame(hof_size)  

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(f"[Main] Starting Evolution on GPU: {config.GENERATIONS} Gens...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=config.GENERATIONS, 
                        stats=stats, halloffame=hof, verbose=True)

    # 4. 保存结果
    utils.save_best_factors(hof, data, toolbox)

if __name__ == "__main__":
    main()