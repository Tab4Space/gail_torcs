# https://github.com/shareeff/PPO/blob/master/worker.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os

from gym_torcs import TorcsEnv
from ppo_agent import PPOAgent
from discriminator import Discriminaor

# hyper parameters
NUM_EPISODE = 6000
GAMMA = 0.9
ACTOR_LR = 0.0001
CRITIC_LR = 0.0002
BATCH_SIZE = 64
ACTOR_UPDATE_STEP = 15
CRITIC_UPDATE_STEP = 15
STATE_DIM = 29
ACTION_DIM = 3
CLIP_VALUE = 0.2


tf.reset_default_graph()

def convert_obs(obs):
    state = np.hstack((
        obs.angle, obs.track, obs.trackPos, obs.speedY, obs.speedX, obs.speedZ, obs.wheelSpinVel/100.0, obs.rpm
    )).reshape(1, STATE_DIM)
    return state


def main():
    # 전문가 데이터 load
    # expert_states = np.genfromtxt('./observation_ddpg.csv', delimiter=',', dtype=np.float32)
    # expert_actions = np.genfromtxt('./action_ddpg.csv', delimiter=',', dtype=np.float32)

    expert_states = np.load('./expert_state.npy')
    expert_actions = np.load('./expert_action.npy')

    # Env, model load
    env = TorcsEnv(vision=False, throttle=True, text=True, gear_change=False)
    ppo = PPOAgent()
    D = Discriminaor()
    # saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('./save_model10/max_score/max_model_111352.8971768222.ckpt.meta')

    score_buf, graph_d_reward = [], []

    MAX_STEP = 906
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './save_model10/max_score/max_model_111352.8971768222.ckpt')
        max_score, ep_score, max_ep_score, change_count = 0, 0, 0, 0

        for ep in range(NUM_EPISODE):
            action_buf, state_buf, reward_buf = [], [], []
            step, score =0, 0
            done = False
            ep_score = 0
            # memory 때문에 20번마다 한번씩 relaunch
            if np.mod(ep, 20) == 0: obs = env.reset(relaunch=True)
            else:   obs = env.reset()

            state = convert_obs(obs)

            while not (step == MAX_STEP):
                if done:
                    print('\nDone: {0}\n'.format(ep_score))
                    
                    if ep_score > max_ep_score and ep_score > 25000 and MAX_STEP==302:
                        max_score = score
                        saver.save(sess, './save_model20/per_episode/epMAX_'+str(step)+'_'+str(ep_score)+'.ckpt')
                        print('\n########## update max score and save model #########\n')

                    obs = env.reset()
                    state = convert_obs(obs)
                    max_ep_score = ep_score
                    ep_score = 0

                step += 1

                action = ppo.choose_action(state)
                next_obs, reward, done, _ = env.step(action)
                
                state_buf.append(state)
                action_buf.append(action)
                reward_buf.append(reward)

                score += reward
                ep_score += reward
                next_state = convert_obs(next_obs)
                state = next_state
                print('\r{}/{}'.format(step, MAX_STEP), flush=True, end='')

            score_buf.append(score)
            
            # Discriminator Train
            for _ in range(2):
                # sample_indices = (np.random.randint(low=0, high=expert_states.shape[0], size=MAX_STEP))
                # inp = [expert_states, expert_actions]
                # sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data


                start_idx = np.random.randint(low=0, high=4799)
                start_idx = start_idx * 280

                sampled_expert_s = expert_states[start_idx:start_idx+len(state_buf), :]
                sampled_expert_a = expert_actions[start_idx:start_idx+len(action_buf), :]

                D.train(expert_s=sampled_expert_s, expert_a=sampled_expert_a, agent_s=np.vstack(state_buf), agent_a=np.vstack(action_buf))

            d_rewards = D.get_rewards(agent_s=np.vstack(state_buf), agent_a=np.vstack(action_buf))
            d_reward_buf = [np.asscalar(r) for r in d_rewards]
            graph_d_reward.append(sum(d_reward_buf))
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

            # if score > max_score and score > 90000:
            #     max_score = score
            #     saver.save(sess, './save_model17/max_score/max_model_'+str(max_score)+'.ckpt')
            #     print('\n########## update max score and save model #########\n')

            if ep % 50== 0 and ep > 0:
                fig = plt.figure(figsize=(16, 8))
                plt.xlabel('EP')
                plt.ylabel('SCORE')
                plt.plot(list(range(len(score_buf))), score_buf, c='r', lw=1, ls='-')
                fig.savefig('./save_model20/graph/env_reward_graph.png')
                fig.clear()
                plt.clf()
            if ep % 50== 0 and ep > 0:
                fig = plt.figure(figsize=(16, 8))
                plt.xlabel('EP')
                plt.ylabel('SCORE')
                plt.plot(list(range(len(graph_d_reward))), graph_d_reward, c='b', lw=1, ls='-')
                fig.savefig('./save_model20/graph/d_reward_graph.png')
                fig.clear()
                plt.clf()
                print('\n@@@@@@@@@@ save model(per 200 ep) @@@@@@@@@@\n')

            print('\nEp: {0}\tScore(Env): {1:.6}\tReward(D): {2:.6}\tStep: {3}\n'.format(ep, score, sum(d_reward_buf), step))

            if score > 90000 and MAX_STEP == 906:
                change_count += 1
                if change_count == 100:
                    MAX_STEP = 604
                    ep_score, max_ep_score = 0, 0
            elif score > 60000 and MAX_STEP == 604:
                change_count += 1
                if change_count == 200:
                    MAX_STEP = 302
                    ep_score, max_ep_score = 0, 0

            
    os.system('pkill torcs')


if __name__ == '__main__':
    main()