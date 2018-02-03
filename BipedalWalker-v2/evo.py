import random
import numpy as np
from operator import attrgetter

n_gene   = 100   # The number of genes.
n_ind    = 300   # The number of individuals in a population.
CXPB     = 0.5   # The probability of crossover.
MUTPB    = 0.2   # The probability of individdual mutation.
MUTINDPB = 0.05  # The probability of gene mutation.
NGEN     = 40    # The number of generation loop.

class Individual(np.ndarray):
    """Container of a individual."""
    def __new__(cls, a):
        cls.fitness = None
        return np.asarray(a).view(cls)#Individualオブジェクトとして出力して返す

def create_ind(n_gene):
    """Create a individual."""
    return Individual([random.randint(0, 1) for i in range(n_gene)])

def create_pop(n_ind, n_gene):
    """Create a population."""
    return [create_ind(n_gene) for i in range(n_ind)]

def set_fitness(eval_func, pop):
    """Set fitnesses of each individual in a population."""
    for i, fit in enumerate(map(eval_func, pop)):
        pop[i].fitness = fit

def randomChoice(pop, n_ind, tournsize):
    """Selection function."""
    chosen = []
    for i in range(n_ind):
        aspirants = [random.choice(pop) for j in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter("fitness")))
    return chosen

def cxTwoPointCopy(ind1, ind2):
    """Crossover function."""
    size = len(ind1)
    tmp1 = ind1.copy()
    tmp2 = ind2.copy()
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size-1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    tmp1[cxpoint1:cxpoint2], tmp2[cxpoint1:cxpoint2] = tmp2[cxpoint1:cxpoint2].copy(), tmp1[cxpoint1:cxpoint2].copy()
    return tmp1, tmp2

def mutFlipBit(ind, indpb):
    """Mutation function."""
    tmp = ind.copy()
    for i in range(len(ind)):
        if random.random() < indpb:
            tmp[i] = type(ind[i])(not ind[i])
    return tmp

if __name__ == "__main__":
    random.seed(1)
    # --- Step1 : Create initial generation.
    pop = create_pop(n_ind, n_gene)
    print(pop)
    set_fitness(sum, pop)
    best_ind = max(pop, key=attrgetter("fitness"))
    # --- Generation loop.
    print("Generation loop start.")
    print("Generation: 0. Best fitness: " + str(best_ind.fitness))
    for g in range(NGEN):
        # --- Step2 : Selection.
        offspring = randomChoice(pop, n_ind, tournsize=3)
        # --- Step3 : Crossover.
        crossover = list()
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1, child2 = cxTwoPointCopy(child1, child2)
                child1.fitness = None
                child2.fitness = None
            crossover.append(child1)
            crossover.append(child2)

        offspring = crossover[:]

        # --- Step4 : Mutation.
        mutant = []
        for mut in offspring:
            if random.random() < MUTPB:
                mut = mutFlipBit(mut, indpb=MUTINDPB)
                mut.fitness = None
            mutant.append(mut)

        offspring = mutant[:]

        # --- Update next population.
        pop = offspring[:]
        set_fitness(sum, pop)

        # --- Print best fitness in the population.
        best_ind = max(pop, key=attrgetter("fitness"))
        print("Generation: " + str(g+1) + ". Best fitness: " + str(best_ind.fitness))

    print("Generation loop ended. The best individual: ")
    print(best_ind)