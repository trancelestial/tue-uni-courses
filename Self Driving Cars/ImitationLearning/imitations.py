import os
import numpy as np
import gym
from pyglet.window import key

import fnmatch
import collections
import time

def load_imitations(data_folder):
    """
    1.1 a)
    Given the folder containing the expert imitations, the data gets loaded and
    stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    act = {}
    obs = {}

    for file in os.listdir(data_folder):
        if fnmatch.fnmatch(file, '*action*'):
            act[file.split('_')[-1]] = np.load(os.path.join(data_folder, file))
        elif fnmatch.fnmatch(file, '*observation*'):
            obs[file.split('_')[-1]] = np.load(os.path.join(data_folder, file))
        else:
            print(f'Unrecognized file: {file}')

    act_sorted = collections.OrderedDict(sorted(act.items(), key=lambda t: t[0]))
    actions = list(act_sorted.values())

    obs_sorted = collections.OrderedDict(sorted(obs.items(), key=lambda t: t[0]))
    observations = list(obs_sorted.values())

    return observations, actions

def save_imitations(data_folder, actions, observations, start_index):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_imitations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    assert(len(actions) == len(observations))

    for i in range(len(actions)):
        np.save(os.path.join(data_folder, f'action_{start_index+i:05d}.npy'), actions[i])
        np.save(os.path.join(data_folder, f'observation_{start_index+i:05d}.npy'), observations[i])


class ControlStatus:
    """
    Class to keep track of key presses while recording imitations.
    """
    '''We changed controls from arrow to WASD and added start recording control on Q'''
    def __init__(self):
        self.start = False
        self.stop = False
        self.save = False
        self.quit = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def key_press(self, k, mod):
        if k == key.Q: self.start = True
        if k == key.ESCAPE: self.quit = True
        if k == key.SPACE: self.stop = True
        if k == key.LSHIFT: self.save = True
        if k == key.A: self.steer = -1.0
        if k == key.D: self.steer = +1.0
        if k == key.W: self.accelerate = 0.5
        if k == key.S: self.brake = 0.8

    def key_release(self, k, mod):
        if k == key.A and self.steer < 0.0: self.steer = 0.0
        if k == key.D and self.steer > 0.0: self.steer = 0.0
        if k == key.W: self.accelerate = 0.0
        if k == key.S: self.brake = 0.0


def record_imitations(imitations_folder, start_index):
    """
    Function to record own imitations by driving the car in the gym car-racing
    environment.
    imitations_folder:  python string, the path to where the recorded imitations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """
    '''start_index denotes what is the starting index of the names for observations and actions
    '''
    env = gym.make('CarRacing-v0').env
    status = ControlStatus()
    total_reward = 0.0

    index_cumulative = 0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation = env.reset()
        env.render()

        # set the functions to be called on key press and key release
        env.viewer.window.on_key_press = status.key_press
        env.viewer.window.on_key_release = status.key_release

        while not status.stop and not status.save and not status.quit:
            # collect all observations and actions
            if status.start:
                # print(f'Recoding!')
                observations.append(observation.copy())
                actions.append(np.array([status.steer, status.accelerate,
                                         status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            total_reward += reward
            env.render()

        if status.save:
            save_imitations(imitations_folder, actions, observations, start_index + index_cumulative)
            status.save = False
            status.start = False
            index_cumulative += len(observations)
            print(f'--------> Next start index should be: {start_index + index_cumulative}')

        print(f'Observation shape: {np.array(observations).shape}\nActions shape: {np.array(actions).shape}')

        status.stop = False
        env.close()
