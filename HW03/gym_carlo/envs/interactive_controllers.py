import numpy as np

class GoalController: # only for CS237B
    def __init__(self, world, steering_lims=[-0.5,0.5]):
        self.control = 0
        world.visualizer.win.bind("<KeyPress-Up>", self.arrow_up_press)
        world.visualizer.win.bind("<KeyPress-Down>", self.arrow_down_press)
        world.visualizer.win.bind("<KeyPress-Left>", self.arrow_left_press)
        world.visualizer.win.bind("<KeyPress-Right>", self.arrow_right_press)
        
        world.visualizer.win.focus_force() # very impolite... Polite version is focus_set(), but it is not always working
    
    def arrow_up_press(self, event):
        self.control = 1
    def arrow_down_press(self, event):
        self.control = 3
    def arrow_left_press(self, event):
        self.control = 0
    def arrow_right_press(self, event):
        self.control = 2

class KeyboardController:
    def __init__(self, world, steering_lims=[-0.5,0.5]):
        self._steering = 0.
        self._throttle = 0.
        
        self.steering_lims = steering_lims
        
        self.min_steering = self.steering_lims[0]
        self.max_steering = self.steering_lims[1]
        
        self.min_throttle = -1.5
        self.max_throttle = +1.5
    
        world.visualizer.win.bind("<KeyRelease-Up>", self.arrow_up_release)
        world.visualizer.win.bind("<KeyRelease-Down>", self.arrow_down_release)
        world.visualizer.win.bind("<KeyRelease-Left>", self.arrow_left_release)
        world.visualizer.win.bind("<KeyRelease-Right>", self.arrow_right_release)
        
        world.visualizer.win.bind("<KeyPress-Up>", self.arrow_up_press)
        world.visualizer.win.bind("<KeyPress-Down>", self.arrow_down_press)
        world.visualizer.win.bind("<KeyPress-Left>", self.arrow_left_press)
        world.visualizer.win.bind("<KeyPress-Right>", self.arrow_right_press)
        
        world.visualizer.win.focus_force() # very impolite... Polite version is focus_set(), but it is not always working
    
    @property
    def steering(self):
        return self._steering
    @property
    def throttle(self):
        return self._throttle
    @steering.setter
    def steering(self, val):
        self._steering = np.clip(val, self.min_steering, self.max_steering)
    @throttle.setter
    def throttle(self, val):
        self._throttle = np.clip(val, self.min_throttle, self.max_throttle)
    
    def arrow_up_release(self, event):
        self.throttle += -1.5
    def arrow_down_release(self, event):
        self.throttle += 1.5
    def arrow_left_release(self, event):
        self.steering += self.steering_lims[0]
    def arrow_right_release(self, event):
        self.steering += self.steering_lims[1]
        
    def arrow_up_press(self, event):
        self.throttle += 1.5
    def arrow_down_press(self, event):
        self.throttle += -1.5
    def arrow_left_press(self, event):
        self.steering += self.steering_lims[1]
    def arrow_right_press(self, event):
        self.steering += self.steering_lims[0]
