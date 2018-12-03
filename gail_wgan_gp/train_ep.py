# GAIL with PPO and WGAN-GP Loss
# Train per episode. -> dynamic steps

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from gym_torcs import TorcsEnv
from ppo_agent import PPOAgent
from discriminator import Discriminaor

NUM_EPISODE = 100000
GAMMA = 0.9
ACTOR_UPDATE_STEP = 15
CRITIC_UPDATE_STEP = 15
STATE_DIM = 29
ACTION_DIM = 3

tf.reset_default_graph()
tf.set_random_seed(1337)
np.random.seed(1337)


def convert_obs(obs):
    state = np.hstack((
        obs.angle, obs.track, obs.trackPos, obs.speedY, obs.speedX, obs.speedZ, obs.wheelSpinVel/100.0, obs.rpm
    )).reshape(1, STATE_DIM)
    return state

def draw_graph(buf, c, path):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(list(range(len(buf))), buf, c=c, lw=1, ls='-')
    fig.savefig(path)
    plt.close(fig)


def main():
    base_path = './save_model13/'
    # 전문가 데이터 load
    # expert_states = np.load('./dataset/expert_state.npy')
    # expert_actions = np.load('./dataset/expert_action.npy')
    expert_states = np.genfromtxt('./dataset/observation_ddpg.csv', delimiter=',', dtype=np.float32)
    expert_actions = np.genfromtxt('./dataset/action_ddpg.csv', delimiter=',', dtype=np.float32)

    env = TorcsEnv(text=True, vision=False, throttle=True, gear_change=False)
    ppo = PPOAgent()
    D = Discriminaor()
    saver = tf.train.Saver(max_to_keep=20)
    score_buf, d_score_buf = [], []

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # saver.restore(sess, './save_model1/models/max_model_52847.298026200726.ckpt')
        saver.restore(sess, './save_model13/models/model_17247.ckpt')
        max_score, success_num = 0, 0

        for ep in range(1, NUM_EPISODE+1):
            done = False
            state_buf, action_buf, reward_buf = [], [], []
            score, step = 0, 0

            if np.mod(ep, 20) == 0:
                obs = env.reset(relaunch=True)
            else:
                obs = env.reset()
            
            state = convert_obs(obs)

            while not done:
                step += 1
                action = ppo.choose_action(state)
                next_obs, reward, done, _ = env.step(action)

                state_buf.append(state)
                action_buf.append(action)
                reward_buf.append(reward)

                score += reward
                next_state = convert_obs(next_obs)
                state = next_state

            score_buf.append(score)

            if score >= 30000:
                success_num += 1
                if success_num >= 30:
                    saver.save(sess, base_path+'early_stop_model.ckpt')
                    break
            else:
                success_num = 0

             # Discriminator Train
            for _ in range(8):
                epsilon = np.random.normal(size=[len(state_buf), 1])
                start_idx = np.random.randint(low=0, high=25)
                start_idx = start_idx * 310

                sampled_expert_s = expert_states[start_idx:start_idx+len(state_buf), :]
                sampled_expert_a = expert_actions[start_idx:start_idx+len(action_buf), :]

                D.train(expert_s=sampled_expert_s, expert_a=sampled_expert_a, agent_s=np.vstack(state_buf), agent_a=np.vstack(action_buf), epsilon=epsilon)

            d_rewards = D.get_rewards_agent(agent_s=np.vstack(state_buf), agent_a=np.vstack(action_buf), epsilon=epsilon)
            d_reward_buf = [np.asscalar(r) for r in d_rewards]
            d_score_buf.append(sum(d_reward_buf))

            # d_reward_buf = [sum(x) for x in zip(reward_buf, d_reward_buf)]

            if done:
                last_value = 0.0
            else:
                last_value = ppo.get_value(next_state)
            
            discounted_reward = []
            for r in d_reward_buf[:: -1]:
                last_value = r + GAMMA * last_value
                discounted_reward.append(last_value)

            discounted_reward.reverse()
            batch_action = np.vstack(action_buf)
            batch_state = np.vstack(state_buf)
            batch_discount_reward = np.array(discounted_reward)[:, np.newaxis]
            ppo.update(batch_state, batch_action, batch_discount_reward)

            if score > max_score and score > 15000:
                max_score = score
                saver.save(sess, base_path+'models/model_'+str(max_score).split('.')[0]+'.ckpt')
                print('\n########## update max score and save model #########\n')

            if ep % 50 == 0:
                draw_graph(score_buf, 'r', base_path+'graph/env_reward.png')
                draw_graph(d_score_buf, 'b', base_path+'graph/d_reward.png')

            print('\nEP{0}\tscore:{1}\tstep:{2}\tsuccess:{3}\n'.format(ep, int(score), step, success_num))

        saver.save(sess, base_path+'finish_train.ckpt')

if __name__ == '__main__':
    main()