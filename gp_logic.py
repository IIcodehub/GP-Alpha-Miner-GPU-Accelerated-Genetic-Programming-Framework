import random
from deap import gp
import seeds
import config

def warm_start_init(icls, pset):
    if config.USE_WARM_START and random.random() < 0.2:
        expr_str = random.choice(seeds.seeds_library)
        try:
            tree = gp.PrimitiveTree.from_string(expr_str, pset)
            return icls(tree)
        except Exception as e:
            print(f"[Warning] Seed loading failed for '{expr_str}': {e}")
            return icls(gp.genHalfAndHalf(pset, min_=1, max_=3))
    else:
        return icls(gp.genHalfAndHalf(pset, min_=1, max_=3))


def mutation_structure_constrained(individual, pset):
    """
    点突变 (Point Mutation)
    """
    valid_terminals = [t for t in pset.terminals[object] if isinstance(t, gp.Terminal)]
    if not valid_terminals:
        return individual,

    for i in range(len(individual)):
        node = individual[i]
        if isinstance(node, gp.Terminal):
            # [修改] 使用 config.P_LEAF_MUTATION
            if random.random() < config.P_LEAF_MUTATION:
                new_term = random.choice(valid_terminals)
                individual[i] = new_term
    
    return individual,

def crossover_structure_constrained(ind1, ind2):
    """
    同构交叉 (Homologous Crossover)
    """
    if len(ind1) != len(ind2):
        return ind1, ind2
    
    for i in range(len(ind1)):
        if isinstance(ind1[i], gp.Terminal) and isinstance(ind2[i], gp.Terminal):
            # [修改] 使用 config.P_LEAF_SWAP
            if random.random() < config.P_LEAF_SWAP:
                term1 = ind1[i]
                term2 = ind2[i]
                ind1[i] = term2
                ind2[i] = term1
                
    return ind1, ind2