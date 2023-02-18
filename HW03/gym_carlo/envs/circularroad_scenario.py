import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

from .world import World
from .agents import Car, CircleBuilding, RingBuilding, Painting
from .geometry import Point
from .graphics import Text, Point as pnt # very unfortunate indeed



MAP_WIDTH = 120
MAP_HEIGHT = 120
LANE_WIDTH = 4.4
SIDEWALK_WIDTH = 2.0
NUM_LANE_MARKERS = 50
LANE_MARKER_WIDTH = 0.5
INNER_BUILDING_RADIUS = 30

LANE_MARKER_RADIUS = INNER_BUILDING_RADIUS + LANE_WIDTH + LANE_MARKER_WIDTH / 2.
LANE_MARKER_HEIGHT = np.sqrt(2*(LANE_MARKER_RADIUS**2)*(1-np.cos(np.pi/NUM_LANE_MARKERS))) # approximate the circle with a polygon and then use cosine theorem

PPM = 5 # pixels per meter


class CircularroadScenario(gym.Env):
    def __init__(self, goal=2): # goal is only for compatibility to other scenarios
        self.seed(0) # just in case we forget seeding
        
        self.init_ego = Car(Point(MAP_WIDTH/2. + LANE_MARKER_RADIUS, MAP_HEIGHT/2.), np.pi/2) # it will be reset by reset(), but anyways...
        self.init_ego.velocity = Point(0., 1.)
        self.init_ego.min_speed = 0.
        self.init_ego.max_speed = 30.
        
        # Two variables below will be used only by the automatic data collector
        self.lane_width = LANE_WIDTH
        self.lane_marker_width = LANE_MARKER_WIDTH
        
        self.dt = 0.1
        self.T = np.inf
        
        self.reset()
        
    def reset(self):
        self.world = World(self.dt, width = MAP_WIDTH, height = MAP_HEIGHT, ppm = PPM)
           
        self.ego = self.init_ego.copy()
        rnd_theta  = self.np_random.uniform()*2*np.pi
        rnd_radius = self.np_random.uniform()*(2*LANE_WIDTH + LANE_MARKER_WIDTH - 2*self.ego.size.y) + INNER_BUILDING_RADIUS + self.ego.size.y
        rnd_speed  = 2.5 + self.np_random.uniform()*5
        
        self.ego.center = Point(rnd_radius*np.cos(rnd_theta), rnd_radius*np.sin(rnd_theta)) + Point(MAP_WIDTH/2., MAP_HEIGHT/2.)
        self.ego.heading = np.mod(rnd_theta + np.pi/2., 2*np.pi) # CARLO assumes 90 degrees is +y direction
        self.ego.velocity = Point(rnd_speed*np.cos(self.ego.heading) + 0.01*self.np_random.uniform()-0.005, rnd_speed*np.cos(self.ego.heading) + 0.01*self.np_random.uniform()-0.005) # a little noise is good for data collection
               
        # To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
        self.circle_building = CircleBuilding(Point(MAP_WIDTH/2., MAP_HEIGHT/2.), INNER_BUILDING_RADIUS, 'gray80')
        self.world.add(self.circle_building)
        self.ring_building = RingBuilding(Point(MAP_WIDTH/2., MAP_HEIGHT/2.), INNER_BUILDING_RADIUS + 2*LANE_WIDTH + LANE_MARKER_WIDTH, 1 + np.sqrt((MAP_WIDTH/2.)**2 + (MAP_HEIGHT/2.)**2), 'gray80')
        self.world.add(self.ring_building)
        
        # Let's also add some lane markers on the ground. This is just decorative. Because, why not.
        for theta in np.arange(0, 2*np.pi, 2*np.pi / NUM_LANE_MARKERS):
            dx = LANE_MARKER_RADIUS * np.cos(theta)
            dy = LANE_MARKER_RADIUS * np.sin(theta)
            self.world.add(Painting(Point(MAP_WIDTH/2. + dx, MAP_HEIGHT/2. + dy), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white', heading = theta))

        self.world.add(self.ego)
        
        return self._get_obs()
        
    def close(self):
        self.world.close()
        
    @property
    def observation_space(self):
        low = np.array([INNER_BUILDING_RADIUS, -np.pi, self.ego.min_speed, 0])
        high= np.array([INNER_BUILDING_RADIUS + 2*LANE_WIDTH + LANE_MARKER_WIDTH, np.pi, self.ego.max_speed, 2*np.pi])
        return Box(low=low, high=high)

    @property
    def action_space(self):
        return Box(low=np.array([-0.5,-2.0]), high=np.array([0.5,1.5]))
    
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @property
    def collision_exists(self):
        return self.world.collision_exists()
        
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.ego.set_control(action[0],action[1])
        self.world.tick()
        
        return self._get_obs(), self._get_reward(), self.collision_exists or self.world.t >= self.T, {}
        
    def _get_reward(self): # noone should use this, but let's keep it
        if self.collision_exists:
            return -200
        return 0
        
    def _get_obs(self):
        return np.array([self.ego.distanceTo(Point(MAP_WIDTH/2., MAP_HEIGHT/2.)), np.arctan2(self.ego.y, self.ego.x), self.ego.speed, self.ego.heading])
        
    def car_in_lane(self, lane_no):
        if lane_no == 0:
            d = self.ego.distanceTo(Point(MAP_WIDTH / 2., MAP_HEIGHT / 2.)) - INNER_BUILDING_RADIUS - LANE_WIDTH/2.
        elif lane_no == 1:
            d = self.ego.distanceTo(Point(MAP_WIDTH / 2., MAP_HEIGHT / 2.)) - INNER_BUILDING_RADIUS - 3*LANE_WIDTH/2. - LANE_MARKER_WIDTH
        return np.abs(d) < 1.5
        
    def render(self, mode='rgb'):
        self.world.render()
        
    def write(self, text): # this is hacky, it would be good to have a write() function in world class
        if hasattr(self, 'txt'):
            self.txt.undraw()
        self.txt = Text(pnt(PPM*(MAP_WIDTH/2.), self.world.visualizer.display_height - PPM*10), text)
        self.txt.draw(self.world.visualizer.win)