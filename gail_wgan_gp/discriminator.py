import tensorflow as tf
import numpy as np

NUM_EPISODE = 100000
GAMMA = 0.9
ACTOR_LR = 0.0001
CRITIC_LR = 0.0002
BATCH_SIZE = 64
ACTOR_UPDATE_STEP = 20
CRITIC_UPDATE_STEP = 20
STATE_DIM = 29
ACTION_DIM = 3
CLIP_VALUE = 0.2
MAX_STEP = 1024

class Discriminaor(object):
    def __init__(self):
        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            
            self.expert_s_ph = tf.placeholder(dtype=tf.float32, shape=[None, STATE_DIM], name='expert_state')
            self.expert_a_ph = tf.placeholder(dtype=tf.float32, shape=[None, ACTION_DIM], name='expert_action')
            expert_concat = tf.concat([self.expert_s_ph, self.expert_a_ph], axis=1)
            expert_s_a = expert_concat + tf.random_normal(tf.shape(expert_concat), mean=0.2, stddev=0.1, dtype=tf.float32)
            
            self.agent_s_ph = tf.placeholder(dtype=tf.float32, shape=[None, STATE_DIM], name='agent_state')
            self.agent_a_ph = tf.placeholder(dtype=tf.float32, shape=[None, ACTION_DIM], name='agent_action')
            agent_concat = tf.concat([self.agent_s_ph, self.agent_a_ph], axis=1)
            agent_s_a = agent_concat + tf.random_normal(tf.shape(agent_concat), mean=0.2, stddev=0.1, dtype=tf.float32)

            
            self.epsilon = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='epsilon')
            X_hat_s = self.expert_s_ph + self.epsilon * (self.agent_s_ph - self.expert_s_ph)
            X_hat_a = self.expert_a_ph + self.epsilon * (self.agent_a_ph - self.expert_a_ph)
            X_hat_concat = tf.concat([X_hat_s, X_hat_a], axis=1)

            with tf.variable_scope('network') as network_scope:
                expert_logits = self._build_network_D(expert_s_a)
                network_scope.reuse_variables()  # share parameter
                agent_logits = self._build_network_D(agent_s_a)
                network_scope.reuse_variables()
                X_hat_crit = self._build_network_D(X_hat_concat)

            LAMBDA = 2

            with tf.variable_scope('loss'):
                origin_d_loss = tf.reduce_mean(agent_logits) - tf.reduce_mean(expert_logits)
                gradient_d_X_hat = tf.gradients(X_hat_crit, [X_hat_concat])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient_d_X_hat), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.0)**2)
                loss = origin_d_loss + LAMBDA*gradient_penalty

            optimizer = tf.train.AdamOptimizer(0.00005)
            self.Dtrain_op = optimizer.minimize(loss)
            self.rewards_agent = tf.exp(agent_logits)
            self.rewards_expert = tf.exp(expert_logits)
            self.WGAN_LOSS = loss

    def _build_network_D(self, inputs):
        # Dlayer1 = tf.layers.dense(inputs, 128, activation=tf.nn.tanh, name='Dlayer1')
        # Dlayer2 = tf.layers.dense(Dlayer1, 64, activation=tf.nn.tanh, name='Dlayer2')
        # Dlayer3 = tf.layers.dense(Dlayer2, 64, activation=tf.nn.tanh, name='Dlayer3')
        # Dlayer4 = tf.layers.dense(Dlayer3, 32, activation=tf.nn.tanh, name='Dlayer4')
        # logit = tf.layers.dense(Dlayer4, 1, activation=None, name='Dlogits')
        # return logit

        # xavier init
        Dlayer1 = tf.layers.dense(inputs, 128, activation=tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer(), name='Dlayer1')
        Dlayer2 = tf.layers.dense(Dlayer1, 64, activation=tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer(), name='Dlayer2')
        Dlayer3 = tf.layers.dense(Dlayer2, 64, activation=tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer(), name='Dlayer3')
        Dlayer4 = tf.layers.dense(Dlayer3, 32, activation=tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer(), name='Dlayer4')
        logit = tf.layers.dense(Dlayer4, 1, activation=None, name='Dlogits')
        return logit

    def train(self, expert_s, expert_a, agent_s, agent_a, epsilon):
        feed_dict={
            self.expert_s_ph: expert_s,
            self.expert_a_ph: expert_a,
            self.agent_s_ph: agent_s,
            self.agent_a_ph: agent_a,
            self.epsilon: epsilon
        }
        return tf.get_default_session().run(self.Dtrain_op, feed_dict)

    def get_rewards_agent(self, agent_s, agent_a, epsilon):
        feed_dict={self.agent_s_ph:agent_s, self.agent_a_ph:agent_a, self.epsilon:epsilon}
        return tf.get_default_session().run(self.rewards_agent, feed_dict)

    def get_rewards_expert(self, agent_s, agent_a, epsilon):
        feed_dict={self.agent_s_ph:agent_s, self.agent_a_ph:agent_a, self.epsilon:epsilon}
        return tf.get_default_session().run(self.rewards_expert, feed_dict)

