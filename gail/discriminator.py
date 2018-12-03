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
            
            self.agent_s_ph = tf.placeholder(dtype=tf.float32, shape=[None, STATE_DIM], name='agent_state')
            self.agent_a_ph = tf.placeholder(dtype=tf.float32, shape=[None, ACTION_DIM], name='agent_action')

            expert_concat = tf.concat([self.expert_s_ph, self.expert_a_ph], axis=1)
            expert_s_a = expert_concat + tf.random_normal(tf.shape(expert_concat), mean=0.2, stddev=0.1, dtype=tf.float32)
            agent_concat = tf.concat([self.agent_s_ph, self.agent_a_ph], axis=1)
            agent_s_a = agent_concat + tf.random_normal(tf.shape(agent_concat), mean=0.2, stddev=0.1, dtype=tf.float32)

            with tf.variable_scope('network') as network_scope:
                expert_logits = self._build_network_D(expert_s_a)
                network_scope.reuse_variables()  # share parameter
                agent_logits = self._build_network_D(agent_s_a)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.zeros_like(expert_logits)))
                loss_agent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=agent_logits, labels=tf.ones_like(agent_logits)))
                loss = loss_expert + loss_agent

            optimizer = tf.train.AdamOptimizer(0.0001)
            self.Dtrain_op = optimizer.minimize(loss)
            self.rewards = -tf.log(tf.clip_by_value(agent_logits, 1e-10, 1.0))

    def _build_network_D(self, inputs):
        Dlayer1 = tf.layers.dense(inputs, 128, activation=tf.nn.tanh, name='Dlayer1')
        Dlayer2 = tf.layers.dense(Dlayer1, 64, activation=tf.nn.tanh, name='Dlayer2')
        Dlayer3 = tf.layers.dense(Dlayer2, 32, activation=tf.nn.tanh, name='Dlayer3')
        logit = tf.layers.dense(Dlayer3, 1, name='Dlogits')
        return logit

    def train(self, expert_s, expert_a, agent_s, agent_a):
        feed_dict={
            self.expert_s_ph: expert_s,
            self.expert_a_ph: expert_a,
            self.agent_s_ph: agent_s,
            self.agent_a_ph: agent_a
        }
        return tf.get_default_session().run(self.Dtrain_op, feed_dict)

    def get_rewards(self, agent_s, agent_a):
        feed_dict={self.agent_s_ph:agent_s, self.agent_a_ph:agent_a}
        return tf.get_default_session().run(self.rewards, feed_dict)
