import os
import numpy as np
import random
import math

from gym_unity.envs.unity_env import UnityEnv
from learn import Learner
from learn import ReplayBuffer
#import matplotlib.pyplot as plt
import h5py
from matplotlib import pyplot

import cv2

np.set_printoptions(threshold=np.inf)

# create grid and write to file
maze_trials, max_episode, max_steps = 1, 100000, 900
action_repeat = 30

buffer = ReplayBuffer(10000)
learner = Learner(buffer)

# assume that unity reads that file and generates maze dynamically
env = UnityEnv("/home/gaurav/MySharedRepository/mazeContinuousTarget_fixed_camera/Build/mazeContinuousTarget_fixed_camera",
               0,
               use_visual=True,
               uint8_visual=True)


def drawTrajectory():

    action_repeat = 150

    # size of experience
    #   11 velocities on x axis
    #   11 velocities on y axis
    #   11 actions on x axis
    #   11 actions on y axis
    #   150 action repeats
    #   2 position values


    data_dict = np.zeros((11,11,11,11,150,2))

    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                for l in range(-5, 6):
                    obs_fovea = env.reset()
                    obs_fovea_next, reward, done, info = env.step([[i], [j], [k], [l]])
                    #print(info["brain_info"].vector_observations[0][2])
                    data_dict[i][j][k][l][0][0] = info["brain_info"].vector_observations[0][2]
                    data_dict[i][j][k][l][0][1] = info["brain_info"].vector_observations[0][3]
                    for m in range(1,action_repeat):
                        x_vel_new = info["brain_info"].vector_observations[0][6]
                        y_vel_new = info["brain_info"].vector_observations[0][7]
                        obs_fovea_next, reward, done, info = env.step([[i], [j], [x_vel_new], [y_vel_new]])
                        data_dict[i][j][k][l][m][0] = info["brain_info"].vector_observations[0][2]
                        data_dict[i][j][k][l][m][1] = info["brain_info"].vector_observations[0][3]


    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('dataset_1', data=data_dict)
    h5f.close()

    print("Done")


def runEpisodeActionRepeat():


    obs_fovea = env.reset()
    obs_fovea_next, cummulativeFovealReward, global_done, info = env.step([[0.0], [0.0], [0.0], [0.0]])

    x_pos_new = info["brain_info"].vector_observations[0][2]
    y_pos_new = info["brain_info"].vector_observations[0][3]

    x_pos_prev = x_pos_new
    y_pos_prev = y_pos_new


    #diction1 = np.zeros((2, action_repeat))

    # for steps in max_steps
    for i in range(max_steps):

        #diction1[0][i%action_repeat] = x_pos_new
        #diction1[1][i%action_repeat] = y_pos_new

        # find an action and set a random velocity
        if i % action_repeat == 0:

            if i > 0:

                #pyplot.plot(diction1[0], diction1[1])
                #pyplot.show()
                #diction1 = np.zeros((2, action_repeat))


                # what label -we need to find
                pos_x = x_pos_new-x_pos_prev
                pos_y = y_pos_new-y_pos_prev
                if pos_x < 0:
                    pos_x = math.floor(pos_x)
                else:
                    pos_x = math.ceil(pos_x)
                if pos_y < 0:
                    pos_y = math.floor(pos_y)
                else:
                    pos_y = math.ceil(pos_y)

                true_label = np.zeros((9, 9))

                if 0 <= pos_x+4 < 9:
                    pos_x = pos_x + 4
                else:
                    pos_x = -1
                if 0 <= pos_y+4 < 9:
                    pos_y = pos_y + 4
                else:
                    pos_y = -1

                #print(pos_x, pos_y)

                if pos_x >= 0 and pos_y >= 0:
                    true_label[pos_x][pos_y] = 1.0

                # what velocity -   the initial velocity
                # what tilt -   fixed for a number of steps
                data = np.array([act_x, act_y, init_vel_x, init_vel_y], dtype=np.float32)
                buffer.push(data, true_label.flatten())

                # learn here
                if buffer.__len__() > 1000:
                    learner.learn_step()

                return (data, true_label)

            act_x = random.randrange(-5, 5)
            act_y = random.randrange(-5, 5)

            init_vel_x = random.randrange(-5, 5)
            init_vel_y = random.randrange(-5, 5)

            x_pos_prev = x_pos_new
            y_pos_prev = y_pos_new

            # apply this action and random velocity once
            obs_fovea_next, reward, done, info = env.step([[act_x], [act_y], [init_vel_x], [init_vel_y]])

        x_pos_new = info["brain_info"].vector_observations[0][2]
        y_pos_new = info["brain_info"].vector_observations[0][3]

        x_vel_new = info["brain_info"].vector_observations[0][6]
        y_vel_new = info["brain_info"].vector_observations[0][7]

        # for number of action repeats-1
        # give correct velocity and same action
        obs_fovea_next, reward, done, info = env.step([[act_x], [act_y], [x_vel_new], [y_vel_new]])

    #pyplot.show()
    #pyplot.savefig()

    #print("Test")








def runEpisodeContinuous():

    obs_fovea = env.reset()
    obs_fovea_next, cummulativeFovealReward, global_done, info = env.step([[0.0], [0.0]])

    x_pos_new = info["brain_info"].vector_observations[0][2]
    y_pos_new = info["brain_info"].vector_observations[0][3]
    x_dot_new = info["brain_info"].vector_observations[0][0]
    y_dot_new = info["brain_info"].vector_observations[0][1]

    x_pos_prev = x_pos_new
    y_pos_prev = y_pos_new
    x_pos_prev_1 = x_pos_new
    y_pos_prev_1 = y_pos_new

    x_dot_prev = x_dot_new
    y_dot_prev = y_dot_new
    x_dot_prev_1 = x_dot_new
    y_dot_prev_1 = y_dot_new

    local_buffer = []

    for i in range(max_steps):

        # action repeat here 4-times
        if i % 4 == 0:
            # two random actions
            act_x = random.randrange(-100, 100)/100
            act_y = random.randrange(-100, 100)/100
            x_pos_prev_1 = x_pos_new
            y_pos_prev_1 = y_pos_new
            x_dot_prev_1 = x_dot_new
            y_dot_prev_1 = y_dot_new

        # calculate previous velocities
        x_vel = x_pos_new - x_pos_prev
        y_vel = y_pos_new - y_pos_prev
        x_vel_abs = abs(x_vel)
        y_vel_abs = abs(y_vel)

        # copy new positions to old position: swap history
        x_pos_prev = x_pos_new
        y_pos_prev = y_pos_new
        x_dot_prev = x_dot_new
        y_dot_prev = y_dot_new

        # execute step
        obs_fovea_next, cummulativeFovealReward, global_done, info = env.step([[act_x/10], [act_y/10]])

        # Get new positions
        x_pos_new = info["brain_info"].vector_observations[0][2]
        y_pos_new = info["brain_info"].vector_observations[0][3]
        x_dot_new = info["brain_info"].vector_observations[0][0]
        y_dot_new = info["brain_info"].vector_observations[0][1]

        # build label
        true_label = np.zeros((9,9))
        if x_vel_abs - x_vel_abs/2 <= y_vel_abs <= x_vel_abs + x_vel_abs/2:
            if x_vel >= 0 and y_vel >= 0:
                true_label[3, 5] = 1
            elif x_vel >= 0 and y_vel <= 0:
                true_label[5, 5] = 1
            elif x_vel <= 0 and y_vel >= 0:
                true_label[3, 3] = 1
            elif x_vel <= 0 and y_vel <= 0:
                true_label[5, 3] = 1
        else:
            if x_vel_abs < y_vel_abs:
                if y_vel >= 0:
                    true_label[3, 4] = 1
                elif y_vel < 0:
                    true_label[5, 4] = 1
            elif x_vel_abs >= y_vel_abs:
                if x_vel >= 0:
                    true_label[4, 3] = 1
                elif x_vel < 0:
                    true_label[4, 5] = 1

        true_label = true_label.flatten()
        # save the values in local buffer

        if i % 4 == 0 and i != 0:
            # save data in replay buffer

            # re-calculate velocity from saved prev values
            x_vel_1 = x_pos_new - x_pos_prev_1
            y_vel_1 = y_pos_new - y_pos_prev_1
            # re-calculate cumulative action
            # calculate true-label

            data = np.array([x_vel_1, y_vel_1, x_dot_prev_1 - x_dot_new, y_dot_prev_1 - y_dot_new], dtype=np.float32)
            buffer.push(data, true_label)
            local_buffer = []
        else:
            local_buffer.append(true_label)

        # learn here
        if buffer.__len__() > 1000:
            learner.learn_step()






if __name__ == '__main__':

    drawTrajectory()

    for i in range(max_episode):
        data = runEpisodeActionRepeat()
        if i % 100 == 0:
            learner.plot_loss_graph()
            learner.plot_accuracy_graph()
            learner.plot_sample(str(data[0]), data[1], i)
            learner.plot_prediction(i)
            learner.save_model()
