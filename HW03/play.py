#!/usr/bin/env python3
import numpy as np
import gym_carlo
import gym
import time
import argparse
from gym_carlo.envs.interactive_controllers import KeyboardController
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    args = parser.parse_args()
    scenario_name = args.scenario.lower()
    assert scenario_name in scenario_names, '--scenario argument is invalid!'
    
    env = gym.make(scenario_name + 'Scenario-v0', goal=len(goals[scenario_name]))
    env.seed(int(np.random.rand()*1e6))
    o, d = env.reset(), False
    env.render()
    interactive_policy = KeyboardController(env.world, steering_lims[scenario_name])
    while not d:
        t = time.time()
        a = [interactive_policy.steering, interactive_policy.throttle]
        o,_,d,_ = env.step(a)
        env.render()
        while time.time() - t < env.dt/2: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller