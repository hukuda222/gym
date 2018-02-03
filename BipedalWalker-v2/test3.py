import gym
from gym import wrappers
import time
import random
import pickle
import chainer
from chainer import Function, Variable, optimizers, utils
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import computational_graph as c
from chainer import serializers
import math
import copy
from operator import attrgetter
import pickle

n_gene   = (24*32)+(32*64)+(64*4)   # The number of genes.
n_ind    = 1000   # The number of individuals in a population.
CXPB     = 0.5   # The probability of crossover.
MUTPB    = 0.2   # The probability of individdual mutation.
MUTINDPB = 0.05  # The probability of gene mutation.
NGEN     = 40    # The number of generation loop.

class Q(chainer.Chain):
    def __init__(self ,in_size ,out_size,arr=None):
        if arr==None:
            rand_array = [random.random()*2-1 for i in range(in_size*32+32*64+64*out_size)]
            self.array = copy.deepcopy(rand_array)
        else:
            rand_array = copy.deepcopy(arr)
            self.array = copy.deepcopy(rand_array)

        l1_arr = np.array([[rand_array.pop(-1)for j in range(in_size)]for i in range(32)],dtype=np.float32).astype(np.float32)
        l2_arr = np.array([[rand_array.pop(-1)for j in range(32)]for i in range(64)],dtype=np.float32).astype(np.float32)
        l3_arr = np.array([[rand_array.pop(-1)for j in range(64)]for i in range(out_size)],dtype=np.float32).astype(np.float32)

        super(Q,self).__init__(
            l1=L.Linear(in_size, 32,initialW=l1_arr),
            l2=L.Linear(32, 64,initialW=l2_arr),
            l3=L.Linear(64, out_size,initialW=l3_arr),
        )
    def __call__(self, x, t=None, train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        y = F.leaky_relu(self.l3(h2))
        return y

def randomChoice(popula, tournsize):
    chosen = list()
    for i in range(n_ind):
        aspirants = [random.choice(popula) for j in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter("fitness")).copy())
    return chosen

def get_crossover(ind1, ind2):
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

def mutFlipBit(ind):
    tmp = ind.copy()
    for i in range(len(ind)):
        if random.random() < MUTINDPB:
            tmp[i] = random.random()*2-1
    return tmp

if __name__ == "__main__":
    random.seed(1)
    env = gym.make("BipedalWalker-v2")
    #env = wrappers.Monitor(env, "/tmp/gym-results",force=True)
    observation = env.reset()
    popula = [Q(24,4)for i in range(n_ind)]
    for on in range(NGEN):
        for i in range(n_ind):
            total=0
            while True:
                observation, reward, done, info = env.step(popula[i](np.array([observation],dtype=np.float32).astype(np.float32)).data[0])
                total+=reward
                if done:
                    popula[i].fitness=total
                    print(total)
                    total=0
                    env.reset()
                    break
        print("----------")
        offspring = randomChoice(popula, tournsize=3)
        crossover = list()
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1.array, child2.array = get_crossover(child1.array, child2.array)
                child1.fitness = None
                child2.fitness = None
            crossover.append(child1)
            crossover.append(child2)
        offspring = crossover[:]
        mutant = list()
        for mut in offspring:
            if random.random() < MUTPB:
                mut.array = mutFlipBit(mut.array)
                mut.fitness = None
            mutant.append(mut)
        offspring = mutant[:]
        popula = [Q(24,4,iw.array) for iw in offspring]

    with open('sample.pickle', mode='wb') as f:
        pickle.dump(popula, f, protocol=2)
    env.close()
    #gym.upload("/tmp/gym-results", api_key="sk_W8htz3KcQPyegxf3txMhhw")
