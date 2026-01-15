
import random
import operator
import numpy as np
from functools import partial
from deap import base, creator, tools, gp, algorithms
import warnings
warnings.filterwarnings('ignore')
import gp_logic


import config
import operators
import data_loader
import fitness
import utils

def main():
    # 1. 加载数据 (自动进入 GPU)
    data = data_loader.DataPortalGPU()
    
    # 2. GP 环境设置
    pset = gp.PrimitiveSet("MAIN", 18) 

    pset.renameArguments(
        ARG0='RET',       ARG1='GAP',       ARG2='HL_R', 
        ARG3='CO_R',      ARG4='L_VOL',     ARG5='TO_RATE', 
        ARG6='L_CAP',     ARG7='VWAP_D',    ARG8='AMI', 
        ARG9='BODY',      ARG10='UP_S',     ARG11='LO_S',
        ARG12='LOG_RET',  ARG13='SKEW',     ARG14='KURT', 
        ARG15='BB_WIDTH', ARG16='ATR',      ARG17='VOL_SKEW')
        
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
    pset.addPrimitive(operators.if_else, 3) 
    pset.addPrimitive(operators.step, 1)
    pset.addPrimitive(operators.signed_sq, 1)
    pset.addPrimitive(operators.signed_sqrt, 1)
    pset.addPrimitive(operators.ts_beta_10, 2)
    pset.addPrimitive(operators.ts_resid_10, 2)
    pset.addPrimitive(operators.ts_argmax_10, 1)
    pset.addPrimitive(operators.ts_argmin_10, 1)

    # 4. 注册求和算子
    def ts_sum_10(x): return operators.ts_sum(x, 10)
    pset.addPrimitive(ts_sum_10, 1)

    pset.addPrimitive(operators.ts_corr_10, 2)
    pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

    # Creator
    if hasattr(creator, 'FitnessMax'): del creator.FitnessMax
    if hasattr(creator, 'Individual'): del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Toolbox
    toolbox = base.Toolbox()

    toolbox.register("individual", gp_logic.warm_start_init, creator.Individual, pset)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", partial(fitness.calculate_fitness_gpu, data_portal=data, toolbox=toolbox))
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)

    if config.RESTRICT_STRUCTURE:
        print("[Main] Using Structure-Preserving Point Mutation.")
        toolbox.register("mutate", gp_logic.point_mutation, pset=pset)
    else:
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
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