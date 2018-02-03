import time, math, random, bisect, copy
import gym
from gym import wrappers
import numpy as np
import pickle
import chainer
from chainer import Function, Variable, optimizers, utils
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import computational_graph as c
from chainer import serializers


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu2(x):
    x[x>1]=1
    x[x<-1]=-1
    return x

class NeuralNet():
    def __init__(self, nodeCount):
        self.fitness = 0.0
        #self.nodeCount = np.array([nodeCount],dtype=np.float32))#ニューロンの数
        self.nodeCount = nodeCount
        self.weights = []
        self.biases = []
        for i in range(len(nodeCount) - 1):
            self.weights.append( np.random.uniform(low=-2, high=2, size=(nodeCount[i], nodeCount[i+1])) )
            self.biases.append( np.random.uniform(low=-2, high=2, size=(nodeCount[i+1])))
        '''
        super(NeuralNet,self).__init__(
            l1=L.Linear(nodeCount[0], nodeCount[1],
                initialW = self.weights[0],
                initial_bias = self.biases[0]),
            l2=L.Linear(nodeCount[1], nodeCount[2],
                initialW = self.weights[1],
                initial_bias = self.biases[1]),
            l3=L.Linear(nodeCount[2], nodeCount[3],
                initialW = self.weights[2],
                initial_bias = self.biases[2]),
            l4=L.Linear(nodeCount[3], nodeCount[4],
                initialW = self.weights[3],
                initial_bias = self.biases[3]),
        )
        '''
    '''
    def getOutput(self, x, t=None, train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        y = F.leaky_relu(self.l4(h3))

        if train:
            return F.mean_squared_error(y,t)
        else:
            return y.data[0]
    '''
    #forward
    def getOutput(self, input):
        output = input
        for i in range(len(self.nodeCount)-1):
            output = relu2(np.reshape( np.matmul(output, self.weights[i]) + self.biases[i], (self.nodeCount[i+1])))
        return output


class Population :
    #populationCount:世代あたりの数、nodeCount:ニューロンの数
    def __init__(self, populationCount, mutationRate, nodeCount):
        self.nodeCount = nodeCount
        self.popCount = populationCount
        self.m_rate = mutationRate
        self.population = [ NeuralNet(nodeCount) for i in range(populationCount)]


    def createChild(self, nn1, nn2):
        child = NeuralNet(self.nodeCount)
        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    if random.random() > self.m_rate:
                        if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                            child.weights[i][j][k] = nn1.weights[i][j][k]
                        else :
                            child.weights[i][j][k] = nn2.weights[i][j][k]

        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                if random.random() > self.m_rate:
                    if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                        child.biases[i][j] = nn1.biases[i][j]
                    else:
                        child.biases[i][j] = nn2.biases[i][j]

        return child


    def createNewGeneration(self, bestNN):
        nextGen = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.popCount):
            if random.random() < float(self.popCount-i)/self.popCount:
                #そのまま次の世代にいく
                nextGen.append(copy.deepcopy(self.population[i]))

        fitnessSum = [0]
        minFit = min([i.fitness for i in nextGen])
        for i in range(len(nextGen)):
            fitnessSum.append(fitnessSum[i]+(nextGen[i].fitness-minFit)**4)


        while(len(nextGen) < self.popCount):
            #2つの親を決めてる
            r1 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            r2 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            i1 = bisect.bisect_left(fitnessSum, r1)
            i2 = bisect.bisect_left(fitnessSum, r2)
            if 0 <= i1 < len(nextGen) and 0 <= i2 < len(nextGen) :
                nextGen.append( self.createChild(nextGen[i1], nextGen[i2]) )
            else :
                print("Index Error ");
                print("Sum Array =",fitnessSum)
                print("Randoms = ", r1, r2)
                print("Indices = ", i1, i2)
        self.population.clear()
        self.population = nextGen

def replayBestBots(bestNeuralNets, steps, sleep):
    choice = input("Do you want to watch the replay ?[Y/N] : ")
    if choice=='Y' or choice=='y':
        for i in range(len(bestNeuralNets)):
            if (i+1)%steps == 0 :
                observation = env.reset()
                totalReward = 0
                for step in range(MAX_STEPS):
                    env.render()
                    time.sleep(sleep)
                    action = bestNeuralNets[i].getOutput(observation)
                    observation, reward, done, info = env.step(action)
                    totalReward += reward
                    if done:
                        observation = env.reset()
                        break
                print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))


def recordBestBots(bestNeuralNets):
    print("\n Recording Best Bots ")
    print("---------------------")
    env.monitor.start('Artificial Intelligence/'+GAME, force=True)
    observation = env.reset()
    for i in range(len(bestNeuralNets)):
        totalReward = 0
        for step in range(MAX_STEPS):
            env.render()
            action = bestNeuralNets[i].getOutput(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                observation = env.reset()
                break
        print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))
    env.monitor.close()


def uploadSimulation():
    API_KEY = open('/home/dollarakshay/Documents/API Keys/Open AI Key.txt', 'r').read().rstrip()
    gym.upload('Artificial Intelligence/'+GAME, api_key=API_KEY)



def mapRange(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.

    return rightMin + (valueScaled * rightSpan)

def normalizeArray(aVal, aMin, aMax):
    res = []
    for i in range(len(aVal)):
        res.append( mapRange(aVal[i], aMin[i], aMax[i], -1, 1) )
    return res

def scaleArray(aVal, aMin, aMax):
    res = []
    for i in range(len(aVal)):
        res.append( mapRange(aVal[i], -1, 1, aMin[i], aMax[i]) )
    return res

GAME = 'BipedalWalker-v2'
MAX_STEPS = 1000
MAX_GENERATIONS = 1000
POPULATION_COUNT = 200
MUTATION_RATE = 0.01
env = gym.make(GAME)
observation = env.reset()
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.shape[0]
obsMin = env.observation_space.low
obsMax = env.observation_space.high
actionMin = env.action_space.low
actionMax = env.action_space.high
pop = Population(POPULATION_COUNT, MUTATION_RATE, [in_dimen, 8, 16, 8, out_dimen])
bestNeuralNets = []


print("\nObservation\n--------------------------------")
print("Shape :", in_dimen, " \n High :", obsMax, " \n Low :", obsMin)
print("\nAction\n--------------------------------")
print("Shape :", out_dimen, " | High :", actionMax, " | Low :", actionMin,"\n")

for gen in range(MAX_GENERATIONS):
    genAvgFit = 0.0
    minFit =  1000000
    maxFit = -1000000
    maxNeuralNet = None
    for nn in pop.population:
        observation = env.reset()
        totalReward = 0
        for step in range(MAX_STEPS):
            #env.render()
            action = nn.getOutput(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                break

        nn.fitness = totalReward
        minFit = min(minFit, nn.fitness)
        genAvgFit += nn.fitness
        if nn.fitness > maxFit :
            maxFit = nn.fitness
            maxNeuralNet = copy.deepcopy(nn);

    bestNeuralNets.append(maxNeuralNet)
    genAvgFit/=pop.popCount
    print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  " % (gen+1, minFit, genAvgFit, maxFit) )
    pop.createNewGeneration(maxNeuralNet)
    with open('sample.pickle', mode='wb') as f:
        pickle.dump(bestNeuralNets, f)


#recordBestBots(bestNeuralNets)

#uploadSimulation()

#replayBestBots(bestNeuralNets, max(1, int(math.ceil(MAX_GENERATIONS/10.0))), 0.0625)
