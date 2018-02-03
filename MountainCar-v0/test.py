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
_mini_size = 100

# DQN内部で使われるニューラルネット
class Q(chainer.Chain):
    def __init__(self ,in_size ,out_size):
        super(Q,self).__init__(
            l1=L.Linear(in_size, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 100),
            l4=L.Linear(100, out_size, initialW=np.zeros((out_size, 100), dtype=np.float32))
        )

    def __call__(self, x, t=None, train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        y = self.l4(h3)

        if train:
            return F.mean_squared_error(y,t)
        else:
            return y

class DQN:
    def __init__(self,e=1.0):
        self.model = Q(2,3)
        #serializers.load_npz("model.npz", self.model)
        self.model_cp = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.e=e
        self.gamma=0.99
        self.last_move=None
        self.last_pred=[0 for i in range(3)]
        self.pos=[0 for i in range(2)]
        self.batch=[list() for i in range(2)]
        self.step=0
        self.max=-0.6

    def act(self):
        x=np.array([self.pos],dtype=np.float32).astype(np.float32)
        pred=self.model_cp(x)
        self.last_pred = np.asanyarray(copy.deepcopy(pred.data[0,:]))
        #print(self.last_pred)
        act=np.argmax(pred.data,axis=1)[0]
        #print(act)
        if random.random() < self.e:
            act = random.randint(0,2)

        self.last_move=act
        return act

    def reset(self):
        self.last_move=0
        self.last_pred=[0 for i in range(2)]
        self.pos=[0 for i in range(2)]
        self.max=-0.6

    #pが報酬
    def Learn_tank(self,pos,end,p):
        self.step+=1
        if self.max < pos[0]:
            self.max=pos[0]
        if end:
            maxQnew=0
        else:
            x=np.array([pos],dtype=np.float32).astype(np.float32)
            #maxQnewは次の手で一番報酬が高い手の報酬
            maxQnew=np.max(self.model_cp(Variable(x)).data[0])
        update = p + (self.gamma*maxQnew)
        self.last_pred[self.last_move]=update
        self.batch[0].append(self.pos)#np.array(self.pos,dtype=np.float32).astype(np.float32))
        self.batch[1].append(self.last_pred)#np.array(self.last_pred,dtype=np.float32).astype(np.float32))
        #x=np.array([pos],dtype=np.float32).astype(np.float32)
        #t=np.array([self.last_pred],dtype=np.float32).astype(np.float32)
        if len(self.batch[0]) >= _batch_size:
            self.Learn()
        self.pos=pos

    def Learn(self):
        if self.e > 0.0 and self.step>1000:
            self.e -= 0.005
        rand_batch = [list(),list()]
        for i in np.random.permutation(range(len(self.batch[0]))):
            rand_batch[0].append(self.batch[0][i])
            rand_batch[1].append(self.batch[1][i])
        rand_batch[0]=np.array(rand_batch[0],dtype=np.float32).astype(np.float32)
        rand_batch[1]=np.array(rand_batch[1],dtype=np.float32).astype(np.float32)
        for i in range(len(rand_batch[0]))[::_mini_size]:
            self.model.zerograds()
            self.model(Variable(rand_batch[0][i:i+_mini_size]),Variable(rand_batch[1][i:i+_mini_size]),train=True).backward()
            self.optimizer.update()
        self.batch[0]=list()
        self.batch[1]=list()
        self.model_cp = copy.deepcopy(self.model)
        '''
        for i in np.random.permutation(range(len(self.batch[0]))):
            self.model.zerograds()
            self.model(self.batch[0][i],self.batch[1][i],train=True).backward()
            self.optimizer.update()
        self.batch[0]=list()
        self.batch[1]=list()
        self.model_cp = copy.deepcopy(self.model)
        '''

if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    #env = wrappers.Monitor(env, "/tmp/gym-results",force=True)
    dqn = DQN()
    dqn.pos = env.reset()
    count = 0
    po=0
    while True:
        #env.render()
        observation, reward, done, info = env.step(dqn.act())
        #print(env.action_space.sample())
        #print(observation, reward, done, info)
        dqn.Learn_tank(observation,done,reward)
        po+=reward
        if done:
            count+=1
            print(po)
            po=0
            if count > 1000:
                break
            dqn.reset()
            dqn.pos=env.reset()
    #serializers.save_npz("model.npz", dqn.model)
    env.close()
    #gym.upload("/tmp/gym-results", api_key="sk_W8htz3KcQPyegxf3txMhhw")
