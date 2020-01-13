import os
from tqdm import trange
from gym_unity.envs.unity_env import UnityEnv
import numpy as np
from numpy import genfromtxt

from learn import Learner
import matplotlib.pyplot as plt

import random
import time
np.set_printoptions(threshold=np.inf)

import math


# create grid and write to file
maze_trials, max_episode, max_steps = 1, 1000000, 2

learner = Learner()
# assume that unity reads that file and generates maze dynamically
env = UnityEnv("/home/gaurav/MySharedRepository/mazeContinuousTarget_fixed_camera/Build/mazeContinuousTarget",
               0,
               use_visual=True,
               uint8_visual=True)

label = np.zeros((9, 9))
label[3:6, 3:6] = 1/9
label = np.expand_dims(label.flatten(), axis=0)


def runEpisodeContinuous():

    obs_fovea = env.reset()
    obs_fovea_next, cummulativeFovealReward, global_done, info = env.step([[0.0], [0.0]])

    pos_x_prev = info["brain_info"].vector_observations[0][2]
    pos_y_prev = info["brain_info"].vector_observations[0][3]

    x_dot_prev = info["brain_info"].vector_observations[0][0]
    y_dot_prev = info["brain_info"].vector_observations[0][1]

    for i in range(max_steps):

        # two random points
        act_x = random.randrange(-100, 100)/100
        act_y = random.randrange(-100, 100)/100

        obs_fovea_next, cummulativeFovealReward, global_done, info = env.step([[act_x/10], [act_y/10]])

        pos_x_new = info["brain_info"].vector_observations[0][2]
        pos_y_new = info["brain_info"].vector_observations[0][3]

        x_dot_new = info["brain_info"].vector_observations[0][0]
        y_dot_new = info["brain_info"].vector_observations[0][1]

        #print([x_dot_prev, x_dot_new, y_dot_prev, y_dot_new, x_dot_prev, x_dot_new, y_dot_prev, y_dot_new])
        #print([x_dot_prev, y_dot_prev, x_dot_new, y_dot_new, pos_x_prev, pos_y_prev, pos_x_new, pos_y_new])

        # derive a view of movement
        true_label = np.zeros((9,9))

        #y = (y_dot_new - y_dot_prev)*(1 - x_dot_new)/(x_dot_new - x_dot_prev) + y_dot_new
        theta = math.atan((y_dot_prev - y_dot_new) / (x_dot_prev - x_dot_new))

        #print(theta)

        if x_dot_new - x_dot_prev >= 0:
            if theta > -1.57 and theta < -1.18:
                #print("0, -1")
                true_label[4,3] = 1
            if theta > -1.18 and theta < -0.39:
                #print("1, -1")
                true_label[5, 3] = 1
            if theta > -0.39 and theta < 0.39:
                #print("1, 0")
                true_label[5, 4] = 1
            if theta > 0.39 and theta < 1.18:
                #print("1, 1")
                true_label[5, 5] = 1
            if theta > 1.18 and theta < 1.57:
                #print("0, 1")
                true_label[4,5] = 1
        elif x_dot_new - x_dot_prev <= 0:
            if theta > -1.57 and theta < -1.18:
                #print("0, -1")
                true_label[4, 3] = 1
            if theta > -1.18 and theta < -0.39:
                #print("-1, -1")
                true_label[3, 3] = 1
            if theta > -0.39 and theta < 0.39:
                #print("-1, 0")
                true_label[3, 4] = 1
            if theta > 0.39 and theta < 1.18:
                #print("-1, 1")
                true_label[3, 5] = 1
            if theta > 1.18 and theta < 1.57:
                #print("-1, 0")
                true_label[3,4] = 1

        true_label = np.expand_dims(true_label.flatten(), axis=0)

        # learn here
        data = np.expand_dims(np.array([x_dot_prev, y_dot_prev, x_dot_prev-x_dot_new, y_dot_prev-y_dot_new], dtype=np.float32), axis=0)
        learner.learn_step(data, true_label)

        # learn here
        data = np.expand_dims(np.array([x_dot_prev, y_dot_prev, 0, 0], dtype=np.float32), axis=0)
        learner.learn_step(data, label)

        obs_fovea = obs_fovea_next

        pos_x_prev = pos_x_new
        pos_y_prev = pos_y_new

        x_dot_prev = x_dot_new
        y_dot_prev = y_dot_new


if __name__ == '__main__':

    # if model available
    # load it
    if os.path.exists("model.json") and os.path.exists("model.h5"):
        learner.load_model()

    for i in range(max_episode):
        runEpisodeContinuous()
        learner.plot_graph()

        if i % 1000 == 0:

            #print(learner.predict(np.expand_dims(np.array([1, 1, 0, 0], dtype=np.float32), axis=0)))

            plt.plot(learner.predict(np.expand_dims(np.array([1, 1, 0, 0], dtype=np.float32), axis=0)))
            plt.savefig('output'+str(i)+'_'+str(1)+'.png')

            plt.plot(learner.predict(np.expand_dims(np.array([1, -1, 0, 0], dtype=np.float32), axis=0)))
            plt.savefig('output'+str(i)+'_'+str(1)+'.png')

            plt.plot(learner.predict(np.expand_dims(np.array([-1, 1, 0, 0], dtype=np.float32), axis=0)))
            plt.savefig('output'+str(i)+'_'+str(1)+'.png')

            plt.plot(learner.predict(np.expand_dims(np.array([-1, -1, 0, 0], dtype=np.float32), axis=0)))
            plt.savefig('output'+str(i)+'_'+str(1)+'.png')

            #print(learner.predict(np.expand_dims(np.array([1, -1, 0, 0], dtype=np.float32), axis=0)))
            #print(learner.predict(np.expand_dims(np.array([-1, 1, 0, 0], dtype=np.float32), axis=0)))
            #print(learner.predict(np.expand_dims(np.array([-1, -1, 0, 0], dtype=np.float32), axis=0)))

    # save model
    learner.save_model()
