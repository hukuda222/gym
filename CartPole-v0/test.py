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

_batch_size = 1000

# DQN内部で使われるニューラルネット
class Q(chainer.Chain):
    def __init__(self ,in_size ,out_size):
        super(Q,self).__init__(
            l1=L.Linear(in_size, 32),
            l2=L.Linear(32, out_size),
        )

    def __call__(self, x, t=None, train=False):
        h1 = F.leaky_relu(self.l1(x))
        y = F.leaky_relu(self.l2(h1))

        if train:
            return F.mean_squared_error(y,t)
        else:
            return y

class DQN:
    def __init__(self,e=0.9):
        self.model = Q(4,2)
        self.model_cp = copy.deepcopy(self.model)
        #serializers.load_npz("model.npz", self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.e=e
        self.gamma=0.9
        self.last_move=None
        self.history=[-1 for i in range(10)]
        self.last_pred=[0 for i in range(2)]
        self.pos=[0 for i in range(4)]
        self.batch=[list() for i in range(2)]

    def act(self):
        x=np.array([self.pos],dtype=np.float32).astype(np.float32)
        pred=self.model_cp(x)
        self.last_pred = pred.data[0,:]
        #print(self.last_pred)
        act=np.argmax(pred.data,axis=1)[0]
        if self.e > 0.0:
            self.e -= 1/(100000)
        if random.random() < self.e:
            act = random.randint(0,1)

        self.last_move=act
        del(self.history[0])
        self.history.append(act)
        return act

    def reset(self):
        self.last_move=0
        self.history=[-1 for i in range(10)]
        self.last_pred=[0 for i in range(2)]
        self.pos=[0 for i in range(4)]

    #pが報酬
    def Learn_tank(self,pos,end,p,co):
        p=0.0
        if end and co < 200:
            maxQnew=0
            p=-1
        elif co ==200:
            maxQnew=0
            p=0
        else:
            x=np.array([pos],dtype=np.float32).astype(np.float32)
            #maxQnewは次の手で一番報酬が高い手の報酬
            maxQnew=copy.deepcopy(np.max(self.model_cp(x).data[0]))
        update = p + (self.gamma*maxQnew)
        self.last_pred[self.last_move]=update
        self.batch[0].append(np.array([self.pos],dtype=np.float32).astype(np.float32))
        self.batch[1].append(np.array([self.last_pred],dtype=np.float32).astype(np.float32))
        #x=np.array([pos],dtype=np.float32).astype(np.float32)
        #t=np.array([self.last_pred],dtype=np.float32).astype(np.float32)
        if len(self.batch[0])>_batch_size:
            self.Learn()
        self.pos=pos

    def Learn(self):
        for i in np.random.permutation(range(len(self.batch[0]))):
            self.model.zerograds()
            self.model(self.batch[0][i],self.batch[1][i],train=True).backward()
            self.optimizer.update()
        self.batch[0]=list()
        self.batch[1]=list()
        self.model_cp = copy.deepcopy(self.model)

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, "/tmp/gym-results",force=True)
    observation = env.reset()
    dqn = DQN()
    count = 0
    c = list()
    co=0
    while True:
        #env.render()
        observation, reward, done, info = env.step(dqn.act())
        co+=reward
        dqn.Learn_tank(observation,done,reward,co)
        if done:
            count+=1
            c.append(co)
            print(co)
            co=0
            if count > 10000:
                break
            env.reset()
            dqn.reset()
    serializers.save_npz("model.npz", dqn.model)
    env.close()
    print(c)
    #gym.upload("/tmp/gym-results", api_key="sk_W8htz3KcQPyegxf3txMhhw")
