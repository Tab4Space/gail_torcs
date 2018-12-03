import tensorflow as tf
import numpy as np
import time

from ppo_agent import PPOAgent
from gym_torcs import TorcsEnv

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

def main():
    ppo = PPOAgent()
    env = TorcsEnv(text=True, vision=False, throttle=True, gear_change=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './save_model13/models/model_59557.ckpt')

        for i in range(1000):
            obs = env.reset()
            state = convert_obs(obs)
            score, step = 0, 0

            while True:
                action = ppo.choose_action(state)
                next_obs, reward, done, _ = env.step(action)
                time.sleep(0.05)
                
                next_state = convert_obs(next_obs)

                score += reward
                step += 1

                if done:
                    print(step, score)
                    env.reset()
                    step, score = 0, 0
                else:
                    state = next_state
            
        

if __name__=='__main__':
    main()