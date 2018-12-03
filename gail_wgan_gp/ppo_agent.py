import tensorflow as tf
import numpy as np

GAMMA = 0.9
ACTOR_LR = 0.0001
CRITIC_LR = 0.0002
ACTOR_UPDATE_STEP = 40
CRITIC_UPDATE_STEP = 40
STATE_DIM = 29
ACTION_DIM = 3
CLIP_VALUE = 0.2


class PPOAgent(object):
    def __init__(self):
        self.state_ph = tf.placeholder(tf.float32, [None, STATE_DIM], 'state_ph')

        # critic network part
        with tf.variable_scope('critic'):
            # layer1 = tf.layers.dense(self.state_ph, 64, tf.nn.tanh)
            # layer2 = tf.layers.dense(layer1, 128, tf.nn.tanh)
            # layer3 = tf.layers.dense(layer2, 128, tf.nn.tanh)
            # layer4 = tf.layers.dense(layer3, 64, tf.nn.tanh)
            # self.value = tf.layers.dense(layer4, 1)

            # xavier init
            layer1 = tf.layers.dense(self.state_ph, 64, tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer())
            layer2 = tf.layers.dense(layer1, 128, tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer())
            layer3 = tf.layers.dense(layer2, 128, tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer())
            layer4 = tf.layers.dense(layer3, 64, tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer())
            self.value = tf.layers.dense(layer4, 1)

            # critic loss를 계산하기 위해 필요한 변수
            self.discounted_r_ph = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')
            self.advantage = self.discounted_r_ph - self.value

            # critic loss 계산
            self.critic_loss = tf.reduce_mean(tf.square(self.advantage))

            # critic update
            self.critic_train_op = tf.train.AdamOptimizer(CRITIC_LR).minimize(self.critic_loss)


        # actor network part
        policy, policy_params = self._build_actor_net('policy', trainable=True)
        old_policy, old_policy_params = self._build_actor_net('old_policy', trainable=False)

        with tf.variable_scope('sample_actiopn'):
            # policy에서 normal distribution을 따르는 action을 하나 뽑음
            self.action = tf.squeeze(policy.sample(1))

        with tf.variable_scope('update_old_policy'):
            # old policy는 optimizer를 통해 업데이트 하지 않고 policy가 update한 weight를 할당받아서 weight을 갱신함
            self.update_old_policy = [old_p.assign(p) for p, old_p in zip(policy_params, old_policy_params)]

        # actor loss를 계산하기 위한 변수
        self.action_ph = tf.placeholder(tf.float32, [None, ACTION_DIM], 'action_ph')
        self.advs_ph = tf.placeholder(tf.float32, [None, 1], 'advantages_ph')

        # actor loss 계산
        with tf.variable_scope('actor_loss'):
            with tf.variable_scope('surrogate1'):
                ratio = tf.exp(policy.log_prob(self.action_ph) - old_policy.log_prob(self.action_ph))
                # ratio = policy.prob(self.action_ph) / old_policy.prob(self.action_ph)
                surrogate1 = ratio * self.advs_ph

            with tf.variable_scope('surrogate2'):
                surrogate2 = tf.clip_by_value(ratio, 1-CLIP_VALUE, 1+CLIP_VALUE) * self.advs_ph

            self.actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        # actor train
        with tf.variable_scope('actor_train'):
            self.actor_train_op = tf.train.AdamOptimizer(ACTOR_LR).minimize(self.actor_loss, var_list=policy_params)


    def _build_actor_net(self, name, trainable):
        """ Actor Network를 만드는 함수
            softplus: log(exp(features) + 1)
        """
        with tf.variable_scope(name):
            # layer1 = tf.layers.dense(self.state_ph, 64, tf.nn.tanh, trainable=trainable)
            # layer2 = tf.layers.dense(layer1, 128, tf.nn.tanh, trainable=trainable)
            # layer3 = tf.layers.dense(layer2, 128, tf.nn.tanh, trainable=trainable)
            # layer4 = tf.layers.dense(layer3, 64, tf.nn.tanh, trainable=trainable)
            
            # xavier init
            layer1 = tf.layers.dense(self.state_ph, 64, tf.nn.tanh, trainable=trainable, kernel_initializer=tf.glorot_normal_initializer())
            layer2 = tf.layers.dense(layer1, 128, tf.nn.tanh, trainable=trainable, kernel_initializer=tf.glorot_normal_initializer())
            layer3 = tf.layers.dense(layer2, 128, tf.nn.tanh, trainable=trainable, kernel_initializer=tf.glorot_normal_initializer())
            layer4 = tf.layers.dense(layer3, 64, tf.nn.tanh, trainable=trainable, kernel_initializer=tf.glorot_normal_initializer())

            steer = tf.layers.dense(layer4, 1, tf.nn.tanh, trainable=trainable)
            accel = tf.layers.dense(layer4, 1, tf.nn.relu, trainable=trainable)
            brake = tf.layers.dense(layer4, 1, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.concat([steer, accel, brake], axis=1)

            steer = tf.layers.dense(layer4, 1, tf.nn.softplus, trainable=trainable)
            accel = tf.layers.dense(layer4, 1, tf.nn.softplus, trainable=trainable)
            brake = tf.layers.dense(layer4, 1, tf.nn.softplus, trainable=trainable)
            sigma = tf.concat([steer, accel, brake], axis=1)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params


    def choose_action(self, s):
        # policy actor network에서 action을 뽑음, 나온 action은 normal distribution에서 나온 action임
        # s: state
        return tf.get_default_session().run(self.action, {self.state_ph: s})


    def get_value(self, s):
        # critic network 에서 value를 얻음
        # s: state
        return tf.get_default_session().run(self.value, {self.state_ph: s})[0, 0]

    def update(self, s, a, r):
        """
        s: sampled state
        a: sampled action
        r: sampled discounted reward
        """
        # 먼저 policy의 weight를 old_policy에 할당
        tf.get_default_session().run(self.update_old_policy)
        advs = tf.get_default_session().run(self.advantage, {self.state_ph: s, self.discounted_r_ph: r})
        # 필요하다면 advantages를 아래와 같이 계산
        advs = (advs - advs.mean()) / (advs.std() + 1e-6)

        for _ in range(ACTOR_UPDATE_STEP):
            tf.get_default_session().run(self.actor_train_op, {self.state_ph:s, self.action_ph:a, self.advs_ph: advs})

        for _ in range(CRITIC_UPDATE_STEP):
            tf.get_default_session().run(self.critic_train_op, {self.state_ph:s, self.discounted_r_ph: r})