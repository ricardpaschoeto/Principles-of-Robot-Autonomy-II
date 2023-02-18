import gym
from gym.envs.registration import register


env_name = 'intersectionScenario-v0'
if env_name in gym.envs.registry:
    del gym.envs.registry[env_name]
register(
    id=env_name,
    entry_point='gym_carlo.envs:' + env_name[:-3],
    kwargs={'goal': 3},
)


env_name = 'circularroadScenario-v0'
if env_name in gym.envs.registry:
    del gym.envs.registry[env_name]
register(
    id=env_name,
    entry_point='gym_carlo.envs:' + env_name[:-3],
    kwargs={'goal': 2}, # it doesn't really have a goal
)


env_name = 'lanechangeScenario-v0'
if env_name in gym.envs.registry:
    del gym.envs.registry[env_name]
register(
    id=env_name,
    entry_point='gym_carlo.envs:' + env_name[:-3],
    kwargs={'goal': 2}
)