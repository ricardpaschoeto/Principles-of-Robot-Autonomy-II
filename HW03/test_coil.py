import numpy as np
from train_coil import NN
import gym_carlo
import gym
import time
import argparse
import tensorflow as tf
from gym_carlo.envs.interactive_controllers import GoalController
from utils import *

def controller_mapping(scenario_name, control):
    """Different scenarios have different number of goals, so let's just clip the user input -- also could be done via np.clip"""
    if control >= len(goals[scenario_name]):
        control = len(goals[scenario_name])-1
    return control

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()
    scenario_name = args.scenario.lower()
    assert scenario_name in scenario_names, '--scenario argument is invalid!'
    
    if args.goal.lower() == 'all':
        goal_id = len(goals[scenario_name])
    else:
        goal_id = np.argwhere(np.array(goals[scenario_name])==args.goal.lower())[0,0] # hmm, unreadable
    env = gym.make(scenario_name + 'Scenario-v0', goal=goal_id)
    
    nn_model = NN(obs_sizes[scenario_name],2)
    nn_model.load_weights('./policies/' + scenario_name + '_all_CoIL')

    episode_number = 10 if args.visualize else 100
    success_counter = 0
    env.T = 200*env.dt - env.dt/2. # Run for at most 200dt = 20 seconds
    for _ in range(episode_number):
        env.seed(int(np.random.rand()*1e6))
        obs, done = env.reset(), False
        if args.visualize:
            env.render()
            interactive_policy = GoalController(env.world)
        while not done:
            t = time.time()
            obs = np.array(obs).reshape(1,-1)
            u = controller_mapping(scenario_name, interactive_policy.control) if args.visualize else goal_id
            action = nn_model(obs,u).numpy().reshape(-1)
            obs,_,done,_ = env.step(action)
            if args.visualize: 
                env.render()
                while time.time() - t < env.dt/2: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if done:
                env.close()
                if args.visualize: time.sleep(1)
                if env.target_reached: success_counter += 1
    if not args.visualize: print('Success Rate = ' + str(float(success_counter)/episode_number))
    