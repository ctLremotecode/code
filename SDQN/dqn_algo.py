import numpy as np
import tensorflow as tf
from VEC_env import VECEnv
import time
import pandas as pd
import matplotlib.pyplot as plt
#from state_normalization import StateNormalization

tf.compat.v1.disable_eager_execution()
MAX_EPISODES = 1000
MEMORY_CAPACITY = 2000
BATCH_SIZE = 256

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.1,
            reward_decay=0.001,
            e_greedy=0.95,
            replace_target_iter=200,
            memory_size=MEMORY_CAPACITY,
            batch_size=BATCH_SIZE,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 0.95
        self.learn_step_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, n_features * 2 + 2), dtype=np.float32)
        self._build_net()
        t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.name_scope('hard_replacement'):
            self.target_replace_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        self.s = tf.keras.Input(shape=(self.n_features,), dtype=tf.float32, name='s')
        self.s_ = tf.keras.Input(shape=(self.n_features,), dtype=tf.float32, name='s_')
        self.r = tf.keras.Input(shape=(), dtype=tf.float32, name='r')
        self.a = tf.keras.Input(shape=(), dtype=tf.int32, name='a')
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        with tf.name_scope('eval_net'):
            e1 = tf.keras.layers.Dense(units=45, activation=tf.nn.relu6, kernel_initializer=w_initializer, bias_initializer = b_initializer, name='e1')(self.s)
            e3 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='e3')(e1)
            self.q_eval = tf.keras.layers.Dense(units=self.n_actions, activation=tf.keras.activations.softmax, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='q')(e3)
        with tf.name_scope('target_net'):
            t1 = tf.keras.layers.Dense(units=45, activation=tf.nn.relu6, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t1')(self.s_)
            t3 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t3')(t1)
            self.q_next = tf.keras.layers.Dense(units=self.n_actions, activation=tf.keras.activations.softmax, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t4')(t3)
        with tf.name_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.name_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.name_scope('train'):
            self._train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })
        self.cost_his.append(cost)
        self.learn_step_counter += 1


if __name__ == '__main__':
    env = VECEnv()
    #Normal = StateNormalization()
    np.random.seed(1)
    tf.random.set_seed(1)
    DQN = DeepQNetwork(env.n_actions, env.n_states, output_graph=False)
    t1 = time.time()
    ep_reward_list = []
    MAX_EP_STEPS = env.TS
    T3 = []
    E3 = []
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        j = 0
        ac = np.zeros((env.nVC, env.nSubTask))
        T1 = np.zeros((env.nVC, env.nSubTask))
        E1 = np.zeros((env.nVC, env.nSubTask))
        T2 = np.zeros(env.nVC)
        while j < MAX_EP_STEPS:
            for k in range(env.nVC):
                a = DQN.choose_action(s[k, :])
                ac[k, j] = a
                s_, r, done, Tsub, Esub = env.step(a, s[k, :], j, k, ac)
                T1[k, j] = Tsub
                E1[k, j] = Esub
                if done:
                    continue
                DQN.store_transition(s[k, :], a, r, s_)
                if DQN.memory_counter > MEMORY_CAPACITY:
                    DQN.learn()
                s[k, :] = s_
                ep_reward += r
            j = j+1
        for h in range(env.nVC):
            if env.DAG[h] == 0:
                T2[h] = T1[h, 0]+T1[h, 1]+T1[h, 2]+T1[h, 3]
            else:
                T2[h] = T1[h, 0] + max(T1[h, 1], T1[h, 2]) + T1[h, 3]
        T4 = np.max(T2)
        E4 = np.sum(E1)
        T3 = np.append(T3, T4)
        E3 = np.append(E3, E4)
        ep_reward_list = np.append(ep_reward_list, ep_reward)
    print('Running time: ', time.time() - t1)
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

