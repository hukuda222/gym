import gym
from gym import wrappers
import numpy as np
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
import universe

_batch_size = 100

# DQN内部で使われるニューラルネット
class Q(chainer.Chain):
    def __init__(self ,in_size ,out_size):
        super(Q,self).__init__(
            l1=L.Linear(in_size, 128),
            l2=L.Linear(128, 256),
            l3=L.Linear(256, 512),
            l4=L.Linear(512, out_size),
        )

    def __call__(self, x, t=None, train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.dropout(F.leaky_relu(self.l2(h1)))
        h3 = F.leaky_relu(self.l3(h2))
        y = F.leaky_relu(self.l4(h3))

        if train:
            return F.mean_squared_error(y,t)
        else:
            return y

class DQN:
    def __init__(self,e=0.9):
        self.model = Q(24,4*5)
        #serializers.load_npz("model.npz", self.model)
        self.model_cp = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.e=e
        self.gamma=0.9
        self.last_move=[0 for i in range(4)]
        self.last_pred=[0 for i in range(4*5)]
        self.input=[0 for i in range(24)]
        self.batch=[list() for i in range(2)]
        self.reward=0

    def act(self):
        x=np.array([self.input],dtype=np.float32).astype(np.float32)
        pred=self.model_cp(x)
        self.last_pred = pred.data[0,:]
        act = list()
        for i in range(0,4*5,5):
            act.append((np.argmax(pred.data[0][i:i+4])-2)/2)
        if self.e > 0.5:
            self.e -= 1/(10000000)
        if random.random() < self.e:
            act = [random.randint(-2,2)/4 for i in range(4)]
        self.last_move=act
        return act

    def reset(self):
        self.last_move=[0 for i in range(4)]
        self.last_pred=[0 for i in range(4*5)]
        self.input=[0 for i in range(24)]

    #pが報酬
    def Learn_tank(self,pos,end,p):
        #print(p)
        if end:
            old_target=[0 for i in range(4*5)]
        else:
            x=np.array([pos],dtype=np.float32).astype(np.float32)
            old_target=copy.deepcopy(self.model_cp(x).data[0])
        if abs(p)>0.5:
            p=p*0.5/abs(p)
        elif abs(p)<0.01:
            p=-0.1
        self.reward=0
        for i in range(0,4*5,5):
            self.reward += max(old_target[i:i+4])/4
        for i in range(0,4*5,5):
            self.last_pred[int(i+(self.last_move[int(i/5)]*4)+2)] = p + (self.gamma*self.reward)
            if abs(self.last_pred[int(i+(self.last_move[int(i/5)]*4)+2)]) > 1:
                self.last_pred[int(i+(self.last_move[int(i/5)]*4)+2)]/=abs(self.last_pred[int(i+(self.last_move[int(i/5)]*4)+2)])
        self.batch[0].append(np.array([self.input],dtype=np.float32).astype(np.float32))
        self.batch[1].append(np.array([self.last_pred],dtype=np.float32).astype(np.float32))
        if len(self.batch[0])>_batch_size:
            self.Learn()
        self.input=pos

    def Learn(self):
        for i in np.random.permutation(range(len(self.batch[0]))):
            self.model.zerograds()
            self.model(self.batch[0][i],self.batch[1][i],train=True).backward()
            self.optimizer.update()
        self.batch[0]=list()
        self.batch[1]=list()
        self.model_cp = copy.deepcopy(self.model)

if __name__ == '__main__':
    '''
    #internet.SlitherIO-v0
    env = gym.make("flashgames.DuskDrive-v0")
    #env = wrappers.Monitor(env, "/tmp/gym-results",force=True)
    observation = env.reset()
    #dqn = DQN()
    c = list()
    total=0
    co=0
    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        #dqn.Learn_tank(observation,done,reward)
        total+=reward
        if done:
            c.append(total)
            print(total,reward)
            total=0
            co+=1
            if co%1000==0:
                print("////////////////")
            if len(c) > 10:
                break
            env.reset()
            #dqn.reset()
    #serializers.save_npz("model.npz", dqn.model)
    env.close()
    print(c)
    #gym.upload("/tmp/gym-results", api_key="sk_W8htz3KcQPyegxf3txMhhw")
    '''
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()

    while True:
      action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
      observation_n, reward_n, done_n, info = env.step(action_n)
      env.render()
