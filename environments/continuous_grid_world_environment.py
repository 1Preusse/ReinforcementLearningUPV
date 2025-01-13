import numpy as np

class ContinuousWindyGridworld:
    def __init__(self):
        # Environment bounds
        self.x_min, self.x_max = 0, 10
        self.y_min, self.y_max = 0, 8
        self.start = np.array([0.5, 3.5])
        self.goal = np.array([7.5, 3.5])
        self.wind_strength = [0, 0, 0, 0, 1, 1, 2, 2, 1, 0]
        
        # Define continuous action space bounds
        self.action_space = {
            'radius': (0.0, 5.0),  # Max step size
            'angle': (0.0, 2 * np.pi)
        }
        
        self.state = self.start.copy()
    
    def reset(self):
        """Resets the environment to the start state."""
        self.state = self.start.copy()
        return self.state
    
    def step(self, action):
        """
        Takes an action and returns (next_state, reward, done).
        """
        radius, angle = action
        next_state = self.state + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle)
        ])
        
        # Add wind effect based on x-position
        column = int(np.clip(next_state[0], self.x_min, self.x_max - 1))
        wind_effect = self.wind_strength[column]
        next_state[1] += wind_effect
        
        # Clip to grid bounds
        next_state[0] = np.clip(next_state[0], self.x_min, self.x_max)
        next_state[1] = np.clip(next_state[1], self.y_min, self.y_max)
        
        self.state = next_state
        
        # Calculate reward
        distance_to_goal = np.linalg.norm(self.state - self.goal)
        if distance_to_goal < 0.5:
            reward = 100
            done = True
        else:
            reward = -0.1 * distance_to_goal
            done = False
        
        return self.state, reward, done
