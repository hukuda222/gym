import copy, sys
import numpy as np
from collections import deque
import gym
from gym import wrappers
import random
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers

class Neuralnet(Chain):
    def __init__(self, n_in, n_out):
        super(Neuralnet, self).__init__(
            L1 = L.Linear(n_in, 100),
            L2 = L.Linear(100, 100),
            L3 = L.Linear(100, 100),
            Q_value = L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32))
        )

    def Q_func(self, x):
        h = F.leaky_relu(self.L1(x))
        h = F.leaky_relu(self.L2(h))
        h = F.leaky_relu(self.L3(h))
        h = self.Q_value(h)
        return h

class Agent():
    def __init__(self, n_st, n_act, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)#再帰の深さ上限
        self.n_act = n_act
        self.model = Neuralnet(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.memory = deque()
        self.loss = 0
        self.step = 0
        self.gamma = 0.99
        self.mem_size = 1000
        self.batch_size = 100
        self.epsilon = 1
        self.epsilon_decay = 0.005
        self.epsilon_min = 0
        self.exploration = 1000
        self.train_freq = 10
        self.target_update_freq = 20

    def stock_experience(self, st, act, r, st_dash, ep_end):
        self.memory.append((st, act, r, st_dash, ep_end))
        if len(self.memory) > self.mem_size:
            self.memory.popleft()

    def forward(self, st, act, r, st_dash, ep_end):
        s = Variable(st)
        s_dash = Variable(st_dash)
        Q = self.model.Q_func(s)
        tmp = self.target_model.Q_func(s_dash)
        tmp = list(map(np.max, tmp.data))
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
        for i in range(self.batch_size):
            for j,ac in enumerate(act[i]):
                target[i][ac + j*5] = r[i] + (self.gamma * max_Q_dash[i]) * (not ep_end[i])
        loss = F.mean_squared_error(Q, Variable(target))
        self.loss = loss.data
        return loss

    def suffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    def parse_batch(self, batch):
        st, act, r, st_dash, ep_end = [], [], [], [], []
        for i in range(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
            ep_end.append(batch[i][4])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        ep_end = np.array(ep_end, dtype=np.bool)
        return st, act, r, st_dash, ep_end

    def experience_replay(self):
        mem = self.suffle_memory()
        perm = np.array(range(len(mem)))
        for start in perm[::self.batch_size]:
            index = perm[start:start+self.batch_size]
            batch = mem[index]
            if len(batch) >= self.batch_size:
                st, act, r, st_d, ep_end = self.parse_batch(batch)
                self.model.zerograds()
                loss = self.forward(st, act, r, st_d, ep_end)
                loss.backward()
                self.optimizer.update()

    def get_action(self, st):
        if np.random.rand() < self.epsilon:
            return [random.randint(0, 4) for i in range(4)]
        else:
            s = Variable(st)
            Q = self.model.Q_func(s)
            Q = Q.data[0].reshape(4,5)
            a = list(map(np.argmax,Q))
            return a

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.exploration < self.step:
            self.epsilon -= self.epsilon_decay

    def train(self):
        if len(self.memory) >= self.mem_size:
            if self.step % self.train_freq == 0:
                self.experience_replay()
                self.reduce_epsilon()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def save_model(self, model_dir):
        serializers.save_npz(model_dir + "model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "model.npz", self.model)
if __name__ == "__main__":
    env = gym.make("BipedalWalker-v2")
    #env = wrappers.Monitor(env, "/tmp/gym-results",force=True)
    agent = Agent(24,4*5,113)
    po=0
    for i_episode in range(5000):
        observation = env.reset()
        for t in range(200):
            #env.render()
            state = observation.astype(np.float32).reshape(1,24)
            action = agent.get_action(state)
            input_action = [(ac-2)/2 for ac in action]
            observation, reward, ep_end, _ = env.step(input_action)
            state_dash = observation.astype(np.float32).reshape(1,24)
            agent.stock_experience(state, action, reward, state_dash, ep_end)
            agent.train()
            po+=reward
            if ep_end:
                print(po)
                po=0
                break
    env.close()
    agent.save_model("")
    #gym.upload("/tmp/gym-results", api_key="sk_W8htz3KcQPyegxf3txMhhw")
