import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

from .world import World
from .agents import Car, RectangleBuilding, Pedestrian, Painting
from .geometry import Point
from .graphics import Text, Point as pnt # very unfortunate indeed



MAP_WIDTH = 80
MAP_HEIGHT = 120
LANE_WIDTH = 4.4
SIDEWALK_WIDTH = 2.0
LANE_MARKER_HEIGHT = 3.8
LANE_MARKER_WIDTH = 0.5
BUILDING_WIDTH = (MAP_WIDTH - 2*SIDEWALK_WIDTH - 2*LANE_WIDTH - LANE_MARKER_WIDTH) / 2.

PPM = 5 # pixels per meter

class LanechangeScenario(gym.Env):
    def __init__(self, goal):
        assert 0 <= goal <= 3, 'Undefined goal'
    
        self.seed(0) # just in case we forget seeding
        
        self.active_goal = goal
        
        self.init_ego = Car(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., 0), heading = np.pi/2)
        self.init_ego.velocity = Point(1., 0.)
        self.init_ego.min_speed = 0.
        self.init_ego.max_speed = 30.
        
        self.dt = 0.1
        self.T = 20
        
        self.reset()
        
    def reset(self):
        self.world = World(self.dt, width = MAP_WIDTH, height = MAP_HEIGHT, ppm = PPM)
           
        self.ego = self.init_ego.copy()

        self.ego.center = Point(BUILDING_WIDTH + SIDEWALK_WIDTH + 2 + np.random.rand()*(2*LANE_WIDTH + LANE_MARKER_WIDTH - 4), self.np_random.uniform()* MAP_HEIGHT/10.)
        self.ego.heading += np.random.randn()*0.1
        self.ego.velocity = Point(0, self.np_random.uniform()*10)
       
        self.targets = []
        self.targets.append(Point(BUILDING_WIDTH + SIDEWALK_WIDTH + LANE_WIDTH/2., MAP_HEIGHT))
        self.targets.append(Point(BUILDING_WIDTH + SIDEWALK_WIDTH + 3*LANE_WIDTH/2. + LANE_MARKER_WIDTH, MAP_HEIGHT))
        
        self.world.add(Painting(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, MAP_HEIGHT), 'gray64'))
        self.world.add(Painting(Point(BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, MAP_HEIGHT), 'gray64'))

        # lane markers on the road
        for y in np.arange(LANE_MARKER_HEIGHT/2., MAP_HEIGHT - LANE_MARKER_HEIGHT/2., 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(MAP_WIDTH/2., y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))

        # arrows on the road
        self.world.add(Painting(Point(MAP_WIDTH/2. - LANE_MARKER_WIDTH/2. - LANE_WIDTH/2., MAP_HEIGHT/2.), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., MAP_HEIGHT/2.), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. - LANE_MARKER_WIDTH/2. - LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2.), Point(3*LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2.), Point(3*LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. - LANE_MARKER_WIDTH/2. - LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2. + LANE_MARKER_WIDTH), Point(2*LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2. + LANE_MARKER_WIDTH), Point(2*LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. - LANE_MARKER_WIDTH/2. - LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2. + 2*LANE_MARKER_WIDTH), Point(LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2. + 2*LANE_MARKER_WIDTH), Point(LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))

        self.world.add(RectangleBuilding(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH, MAP_HEIGHT)))
        self.world.add(RectangleBuilding(Point(BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH, MAP_HEIGHT)))
        
        self.world.add(self.ego)
        
        return self._get_obs()
        
    def close(self):
        self.world.close()
        
    @property
    def observation_space(self):
        low = np.array([BUILDING_WIDTH, self.ego.min_speed, 0])
        high= np.array([MAP_WIDTH - BUILDING_WIDTH, self.ego.max_speed, 2*np.pi])
        return Box(low=low, high=high)

    @property
    def action_space(self):
        return Box(low=np.array([-0.5,-2.0]), high=np.array([0.5,1.5]))
    
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def target_reached(self):
        if self.active_goal < len(self.targets):
            return self.targets[self.active_goal].distanceTo(self.ego) < 1.
        return np.min([self.targets[i].distanceTo(self.ego) for i in range(len(self.targets))]) < 1.
    
    @property
    def collision_exists(self):
        return self.world.collision_exists()
        
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.ego.set_control(action[0],action[1])
        self.world.tick()
        
        return self._get_obs(), self._get_reward(), self.collision_exists or self.target_reached or self.world.t >= self.T, {}
        
    def _get_reward(self):
        if self.collision_exists:
            return -200
        if self.active_goal < len(self.targets):
            return -0.01*self.targets[self.active_goal].distanceTo(self.ego)
        return -0.01*np.min([self.targets[i].distanceTo(self.ego) for i in range(len(self.targets))])
        
    def _get_obs(self):
        return np.array([self.ego.center.x, self.ego.speed, self.ego.heading])
        
    def render(self, mode='rgb'):
        self.world.render()
        
    def write(self, text): # this is hacky, it would be good to have a write() function in world class
        if hasattr(self, 'txt'):
            self.txt.undraw()
        self.txt = Text(pnt(PPM*(MAP_WIDTH - BUILDING_WIDTH+2), self.world.visualizer.display_height - PPM*10), text)
        self.txt.draw(self.world.visualizer.win)