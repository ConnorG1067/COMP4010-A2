# COMP 4010 Assignment 2
# Carleton University
# NOTE: This is a sample script to call your functions and get some results. 
#       Change "YourName" to your prefered folder name and put your A2codes.py in the folder.
import time

import ConnorGomes.A2codes as A2codes
from A2helpers import FourRoom, plot_grid_world


def _plotGridWorld():

	goal = 0
	render_mode = "human"  # or "human" to show the demo
	gamma = 0.9
	step_size = 0.1
	epsilon = 0.1
	max_episode = 1000
	max_model_step = 10
	env = FourRoom(render_mode=None, goal=goal)

	### Single run ###
	Pi, q = A2codes.QLearning(env, 
							  gamma=gamma, 
							  step_size=step_size, 
							  epsilon=epsilon, 
							  max_episode=max_episode)
	# Pi, q = A2codes.DynaQ(env, 
	# 						gamma=gamma, 
	# 						step_size=step_size, 
	# 						epsilon=epsilon,
	# 						max_episode=max_episode,
	# 						max_model_step=max_model_step)  # for grads / bonus
	plot_grid_world(env, Pi, q)

	env.close()


def _runExp():

	goal = 0
	env = FourRoom(goal=goal)

	### Multiple runs ###
	# start_time = time.time()
	# results = A2codes.runQLExperiments(env)
	# print(f"Finish runQLExperiments in {time.time() - start_time} seconds")

	start_time = time.time()
	results = A2codes.runDynaQExperiments(env)
	print(f"Finish runDynaQExperiments in {time.time() - start_time} seconds")

	print(results)

	env.close()


if __name__ == "__main__":

	_plotGridWorld()
	_runExp()
