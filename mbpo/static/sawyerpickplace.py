import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
    	obj = next_obs[:, :3]
    	goal = next_obs[:, 3:6]
    	distance = np.linalg.norm(obj - goal, axis=-1)
    	done = distance < 0.1
    	done = done[:, None]
    	return done
