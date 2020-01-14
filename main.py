import os
import numpy as np
import random
import math

from gym_unity.envs.unity_env import UnityEnv
from learn import Learner
from learn import ReplayBuffer
import matplotlib.pyplot as plt
import cv2

np.set_printoptions(threshold=np.inf)

# create grid and write to file
maze_trials, max_episode, max_steps = 1, 100000, 100

buffer = ReplayBuffer(10000)
learner = Learner(buffer)
# assume that unity reads that file and generates maze dynamically
env = UnityEnv("/home/gaurav/MySharedRepository/mazeContinuousTarget_fixed_camera/Build/mazeContinuousTarget_fixed_camera",
               0,
               use_visual=True,
               uint8_visual=True)

label = np.zeros((9, 9))
label[3:6, 3:6] = 1/9
#label = np.expand_dims(label.flatten(), axis=0)
label = label.flatten()


def runEpisodeContinuous():

    obs_fovea = env.reset()
    obs_fovea_next, cummulativeFovealReward, global_done, info = env.step([[0.0], [0.0]])

    x_pos_new = info["brain_info"].vector_observations[0][2]
    y_pos_new = info["brain_info"].vector_observations[0][3]
    x_dot_new = info["brain_info"].vector_observations[0][0]
    y_dot_new = info["brain_info"].vector_observations[0][1]

    x_pos_prev = x_pos_new
    y_pos_prev = y_pos_new
    x_dot_prev = x_dot_new
    y_dot_prev = y_dot_new

    for i in range(max_steps):

        # two random points
        act_x = random.randrange(-100, 100)/100
        act_y = random.randrange(-100, 100)/100

        x_vel = x_pos_new - x_pos_prev
        y_vel = y_pos_new - y_pos_prev

        x_pos_prev = x_pos_new
        y_pos_prev = y_pos_new
        x_dot_prev = x_dot_new
        y_dot_prev = y_dot_new

        obs_fovea_next, cummulativeFovealReward, global_done, info = env.step([[act_x/10], [act_y/10]])

        x_pos_new = info["brain_info"].vector_observations[0][2]
        y_pos_new = info["brain_info"].vector_observations[0][3]
        x_dot_new = info["brain_info"].vector_observations[0][0]
        y_dot_new = info["brain_info"].vector_observations[0][1]

        # derive the direction of movement
        true_label = np.zeros((9,9))
        theta = math.atan((y_pos_prev - y_pos_new) / (x_pos_prev - x_pos_new))
        if x_pos_new - x_pos_prev >= 0:
            if theta > -1.57 and theta < -1.18:
                true_label[4,3] = 1
            elif theta > -1.18 and theta < -0.39:
                true_label[5, 3] = 1
            elif theta > -0.39 and theta < 0.39:
                true_label[5, 4] = 1
            elif theta > 0.39 and theta < 1.18:
                true_label[5, 5] = 1
            elif theta > 1.18 and theta < 1.57:
                true_label[4,5] = 1
        elif x_pos_new - x_pos_prev <= 0:
            if theta > -1.57 and theta < -1.18:
                true_label[4, 3] = 1
            elif theta > -1.18 and theta < -0.39:
                true_label[3, 3] = 1
            elif theta > -0.39 and theta < 0.39:
                true_label[3, 4] = 1
            elif theta > 0.39 and theta < 1.18:
                true_label[3, 5] = 1
            elif theta > 1.18 and theta < 1.57:
                true_label[3,4] = 1

        #true_label = np.expand_dims(true_label.flatten(), axis=0)
        true_label = true_label.flatten()

        print([x_vel, y_vel, x_dot_prev-x_dot_new, y_dot_prev-y_dot_new])



        # learn here
        #data = np.expand_dims(np.array([x_dot_prev, y_dot_prev, x_dot_prev-x_dot_new, y_dot_prev-y_dot_new], dtype=np.float32), axis=0)
        data = np.array([x_vel, y_vel, x_dot_prev-x_dot_new, y_dot_prev-y_dot_new], dtype=np.float32)
        #learner.learn_step(data, true_label)
        # save data in replay buffer
        buffer.push(data, true_label)

        # learn here
        #data = np.expand_dims(np.array([x_dot_prev, y_dot_prev, 0, 0], dtype=np.float32), axis=0)
        #data = np.array([x_dot_prev, y_dot_prev, 0, 0], dtype=np.float32)
        #learner.learn_step(data, label)
        # save data in replay buffer
        #buffer.push(data, label)


        if buffer.__len__() > 100:
            learner.learn_step()


        #obs_fovea = obs_fovea_next



if __name__ == '__main__':

    # if model available
    # load it
    if os.path.exists("model.json") and os.path.exists("model.h5"):
        learner.load_model()

    for i in range(max_episode):
        runEpisodeContinuous()
        learner.plot_graph()
        #learner.evaluate()

        if i % 1000 == 0:

            image = np.reshape(learner.predict( np.expand_dims( np.array([1, 1, 0, 0], dtype=np.float32), axis=0)), (9, 9))
            max_val = np.max(image)
            image = image*255/max_val
            image = cv2.resize(image, (81,81), interpolation=cv2.INTER_AREA)
            cv2.imwrite("evaluation_0_"+str(i)+".png", image)

            image = np.reshape(learner.predict( np.expand_dims( np.array([-1, 1, 0, 0], dtype=np.float32), axis=0)), (9, 9))
            max_val = np.max(image)
            image = image*255/max_val
            image = cv2.resize(image, (81,81), interpolation=cv2.INTER_AREA)
            cv2.imwrite("evaluation_1_"+str(i)+".png", image)

            image = np.reshape(learner.predict( np.expand_dims( np.array([1, -1, 0, 0], dtype=np.float32), axis=0)), (9, 9))
            max_val = np.max(image)
            image = image*255/max_val
            image = cv2.resize(image, (81,81), interpolation=cv2.INTER_AREA)
            cv2.imwrite("evaluation_2_"+str(i)+".png", image)

            image = np.reshape(learner.predict( np.expand_dims( np.array([-1, -1, 0, 0], dtype=np.float32), axis=0)), (9, 9))
            max_val = np.max(image)
            image = image*255/max_val
            image = cv2.resize(image, (81,81), interpolation=cv2.INTER_AREA)
            cv2.imwrite("evaluation_3_"+str(i)+".png", image)

            #image = learner.predict( np.expand_dims( np.array([1, -1, 0, 0], dtype=np.float32), axis=0) )
            #cv2.imwrite("evaluation_1_"+str(i)+".png",image.reshape((9, 9)))

            #image = learner.predict( np.expand_dims( np.array([-1, 1, 0, 0], dtype=np.float32), axis=0) )
            #cv2.imwrite("evaluation_2_"+str(i)+".png",image.reshape((9, 9)))

            #image = learner.predict( np.expand_dims( np.array([-1, -1, 0, 0], dtype=np.float32), axis=0) )
            #cv2.imwrite("evaluation_3_"+str(i)+".png",image.reshape((9, 9)))

            """
            plt.plot(learner.predict(np.expand_dims(np.array([1, -1, 0, 0], dtype=np.float32), axis=0)))
            plt.savefig('output'+str(i)+'_'+str(1)+'.png')
            plt.close()

            plt.plot(learner.predict(np.expand_dims(np.array([-1, 1, 0, 0], dtype=np.float32), axis=0)))
            plt.savefig('output'+str(i)+'_'+str(1)+'.png')
            plt.close()

            plt.plot(learner.predict(np.expand_dims(np.array([-1, -1, 0, 0], dtype=np.float32), axis=0)))
            plt.savefig('output'+str(i)+'_'+str(1)+'.png')
            plt.close()

            print(learner.predict(np.expand_dims(np.array([1, 1, 0, 0], dtype=np.float32), axis=0)))
            print(learner.predict(np.expand_dims(np.array([1, -1, 0, 0], dtype=np.float32), axis=0)))
            print(learner.predict(np.expand_dims(np.array([-1, 1, 0, 0], dtype=np.float32), axis=0)))
            print(learner.predict(np.expand_dims(np.array([-1, -1, 0, 0], dtype=np.float32), axis=0)))
            """

    # save model
    learner.save_model()
