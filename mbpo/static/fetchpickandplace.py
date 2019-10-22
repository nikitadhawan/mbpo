import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
    	achieved_goal = next_obs[:, :3]
    	desired_goal = next_obs[:, 3:6]
    	distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    	done = distance < 0.1
    	done = done[:, None]
    	return done
