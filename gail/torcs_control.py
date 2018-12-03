import sys, time, os
import keyboard
import csv
import numpy as np
import pandas as pd
import argparse

import gym
from gym_torcs import TorcsEnv
import snakeoil3_gym as snakeoil3


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=str)
    return parser.parse_args()

action = np.zeros([3], np.float32)
steer = np.zeros([1], np.float32)
accel = np.zeros([1], np.float32)
brake = np.zeros([1], np.float32)

"""
103 -> up key for accel
105 -> left key for steer left
106 -> right key for steer right
108 -> down key for brake(not use)
57 -> space key for brake(instead of down key)
"""
def get_action(e):
    input_keys = [str(code) for code in keyboard._pressed_events]
    # print(input_keys)
    time.sleep(0.03)

    if len(input_keys) == 0:
        accel[0] -= 0.1
        brake[0] -= 0.1
        steer[0] = 0.0

    if len(input_keys) > 0:
        if '103' in input_keys and len(input_keys) == 1:
            accel[0] += 1e-2*2
            steer[0] = 0.0
            brake[0] -= 0.1
        elif '57' in input_keys and len(input_keys) == 1:
            brake[0] += 1e-2
            steer[0] = 0.0

        elif '105' in input_keys and len(input_keys) == 1:
            steer[0] += 1e-2

        elif '106' in input_keys and len(input_keys) == 1:
            steer[0] -= 1e-2

        # accel + steer left
        elif '103' in input_keys and '105' in input_keys:
            accel[0] -= 1e-2*1.5
            steer[0] += 1e-2

        # accel + steer right
        elif '103' in input_keys and '106' in input_keys:
            accel[0] -= 1e-2*1.5
            steer[0] -= 1e-2

        # accel + brake
        elif '103' in input_keys and '57' in input_keys:
            accel[0] += 1e-2
            brake[0] += 1e-2

        # brake + steer left
        elif '105' in input_keys and '57' in input_keys:
            brake[0] += 1e-2
            steer[0] += 1e-2

        # brake + steer right
        elif '106' in input_keys and '57' in input_keys:
            brake[0] += 1e-2
            steer[0] -= 1e-2


    np.clip(steer, -1.0, 1.0, steer)
    np.clip(accel, 0.0, 1.0, accel)
    np.clip(brake, 0.0, 1.0, brake)

    action[0], action[1], action[2] = steer, accel, brake


def main():
    count = 0
    
    env = TorcsEnv(vision=False, throttle=True, gear_change=False, text=False)
    # action_fd = open('./trajectory/action15.csv', 'w', newline='')
    # observation_fd = open('./trajectory/observation15.csv', 'w', newline='')
    
    # action_writer = csv.writer(action_fd, delimiter=',')
    # observation_writer = csv.writer(observation_fd, delimiter=',')

    for ep in range(1, 5):
        done = False
        step, score = 0, 0
        if np.mod(ep, 5) == 0:
            ob = env.reset(relaunch=True)
            state =  np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            #observation_writer.writerow(state.tolist())

        else:
            ob = env.reset()
            state =  np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            #observation_writer.writerow(state.tolist())

        while not done:
            keyboard.hook(get_action)
            time.sleep(0.05)
            # write action value in action.csv
            
            # observation_writer.writerow(state.tolist())
            # action_writer.writerow([action[0], action[1], action[2]])

            next_ob, reward, done, info = env.step(action)

            next_state =  np.hstack((next_ob.angle, next_ob.track, next_ob.trackPos, next_ob.speedY, next_ob.speedX, next_ob.speedZ, next_ob.wheelSpinVel/100.0, next_ob.rpm))
            #observation_writer.writerow(state.tolist())
            state = next_state
            # print(step, score)
            print(next_ob.lastLapTime)
            score += reward
            step += 1


    action_fd.close()
    observation_fd.close()
    sys.exit()

        
if __name__=='__main__':
    # args = argparser()
    main()