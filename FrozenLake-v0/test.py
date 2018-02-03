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

# DQN内部で使われるニューラルネット
class Q(chainer.Chain):
    def __init__(self ,in_size ,out_size):
        super(Q,self).__init__(
            l1=L.Linear(in_size, 16),
            l2=L.Linear(16, 32),
            l3=L.Linear(32, 64),
            l4=L.Linear(64, 256),
            l5=L.Linear(256, out_size),
        )

    def __call__(self, x, t=None, train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 =  F.leaky_relu(self.l4(h3))
        y = F.leaky_relu(self.l5(h4))

        if train:
            return F.mean_squared_error(y,t)
        else:
            return y

class DQN:
    def __init__(self,e=0.9):
        self.model = Q(2,4)
        #serializers.load_npz("model.npz", self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.e=e
        self.gamma=0.7
        self.last_move=None
        self.history=[-1 for i in range(10)]
        self.last_pred=[0 for i in range(4)]
        self.pos=0
        self.info=0.333

    def act(self):
        x=np.array([[math.floor(self.pos/4),self.pos%4]],dtype=np.float32).astype(np.float32)
        pred=self.model(x)
        self.last_pred = pred.data[0,:]
        #print(self.last_pred)
        act=np.argmax(pred.data,axis=1)[0]
        if self.e > 0.2:
            self.e -= 1/(10000)
        if random.random() < self.e:
            act = random.randint(0,3)

        self.last_move=act
        del(self.history[0])
        self.history.append(act)
        return act

    def reset(self):
        self.last_move=0
        self.history=[-1 for i in range(10)]
        self.last_pred=[0 for i in range(4)]
        self.pos=0

    #pが報酬
    def Learn(self,pos,p,end):
        if self.pos != pos:
            if end:
                maxQnew=0
            else:
                x=np.array([[math.floor(self.pos/4),self.pos%4]],dtype=np.float32).astype(np.float32)
                #now_pred = self.model(x).data[0]
                #self.last_pred = now_pred
                #print(self.model(x).data[0])
                #maxQnewは次の手で一番報酬が高い手の報酬
                maxQnew=np.max(self.model(x).data[0])
            if end and p == 0.0:
                p = -10.0
            if end and p > 0.0:
                p = 10.0
            update = (1-self.gamma)*p + (self.gamma*maxQnew)
            self.last_pred[self.last_move]=update
            x=np.array([[math.floor(self.pos/4),self.pos%4]],dtype=np.float32).astype(np.float32)
            t=np.array([self.last_pred],dtype=np.float32).astype(np.float32)
            self.model.zerograds()
            self.model(x,t,train=True).backward()
            self.optimizer.update()
            self.pos=pos

if __name__ == '__main__':
    env = gym.make("FrozenLake-v0")
    env = wrappers.Monitor(env, "/tmp/gym-results",force=True)
    observation = env.reset()
    dqn = DQN(0)
    count = 0
    c = 0
    while True:
        env.render()
        observation, reward, done, info = env.step(dqn.act())
        dqn.Learn(observation,reward,done)
        print(observation,reward,done)
        if done:
            count+=1
            #print("--------------------------------------------------------------------------------------")
            if reward > 0:
                c+=1
            if count > 1:
                break
            env.reset()
            dqn.reset()
    #serializers.save_npz("model.npz", dqn.model)
    env.close()
    print(c)
    #gym.upload("/tmp/gym-results", api_key="sk_W8htz3KcQPyegxf3txMhhw")
