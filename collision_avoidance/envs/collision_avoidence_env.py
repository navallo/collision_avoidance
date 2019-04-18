from math import sin, cos, sqrt, exp, pi
import time
from random import uniform

import gym
# from gym import error, spaces, utils
from gym.utils import seeding

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import numpy as np
FLAG_DRAW = True
if FLAG_DRAW:
	from tkinter import *
import rvo2

from collision_avoidance.envs.utils import *

#adapting to RLlib
#todo: remove numpy

class Collision_Avoidance_Env(gym.Env, MultiAgentEnv):
	metadata = {'render.modes': ['human']}

	def __init__(self, numAgents = 10):
		self.timeStep = 1/60.
		self.neighborDist = 1.5
		self.maxNeighbors = 5
		self.timeHorizon = 1.5
		self.radius = 0.5  # 2
		self.maxSpeed = 1
		self.velocity = 2#(1, 1)
		self.laser_num = 16
		self.circle_approx_num = 8

		self.numAgents = numAgents

		self.pixelsize = 500 #1000
		self.framedelay = 30
		self.envsize = 10

		self.step_count = 0
		self.max_step = 1000

		#gym parmaters
		#Discrete action space
		# self.action_size = 8
		# self.action_space = spaces.Discrete(self.action_size)

		#Continuous action space
		self.action_space = gym.spaces.Box(low=-self.maxSpeed, high=self.maxSpeed, shape=(2,))
		self.observation_space = gym.spaces.Box(low=-self.neighborDist, high=self.neighborDist, shape=(4 + self.laser_num*4,))

		self.agents_done = [0] * self.numAgents

		# self.seed()

		self._init_comp_laser_rays()
		self._init_comp_circle_approx()

		self.sim = rvo2.PyRVOSimulator(self.timeStep,
									   self.neighborDist, 
									   self.maxNeighbors, 
									   self.timeHorizon, 
									   self.radius, 
									   self.maxSpeed, 
									   self.velocity)
		self._init_world()
		if FLAG_DRAW:
			self._init_visworld()
			# self.draw_update()
		
		self.reset()

	#part of the initial
	def _init_world(self):
		self.world = {}
		self.world["agents_id"] = []
		self.world["obstacles_vertex_ids"] = []
		self.world["laserScans"] = []
		#self.world["laserScans"].append([])  # for now only one laser scan

		self.world["targets_pos"] = []

		#Add agents
		for i in range(self.numAgents):
			pos = (uniform(self.envsize * 0.5, self.envsize), uniform(0, self.envsize))
			angle = uniform(0, 2 * pi)
			pref_vel = (cos(angle), sin(angle))
			self._init_add_agents(pos, pref_vel)

			#set target
			target_pos = (1.0,5.0)
			self.world["targets_pos"].append(target_pos)

		self.update_pref_vel()

		#Add lasers
		for i in range(self.numAgents):
			self.world["laserScans"].append([0.0] * (self.laser_num * 4))

		'''
		#Add lasers
		for i in range(self.numAgents):
			laserScan = []
			for j in range(self.laser_num):
				laserScan.append([0, 0, 0, 0])
			self.world["laserScans"].append(laserScan)

		#why add twice?
		# for i in range(self.laser_num):
			# self.world["laserScans"][0].append([0, 0, 0, 0])
		'''

		#Add an obstacle
		#map wall
		self._init_add_obstacles((-15.0,0.0),(-15.0,self.envsize),(self.envsize,self.envsize),(self.envsize,0.0))
		# self._init_add_obstacles((0.0,0.0),(0.0,self.envsize),(self.envsize,self.envsize),(self.envsize,0.0))

		self._init_add_obstacles((2.0,0.0),(2.5,0.0),(2.5,4.4),(2.0,4.4))
		self._init_add_obstacles((2.0,5.6),(2.5,5.6),(2.5,10.0),(2.0,10.0))
		self.sim.processObstacles()

	def _init_add_agents(self, pos, pref_vel):
		agent_id = self.sim.addAgent(pos, 
									  self.neighborDist, 
									  self.maxNeighbors, 
									  self.timeHorizon, 
									  1.5, 
									  self.radius, 
									  self.maxSpeed, 
									  pref_vel)
		self.world["agents_id"].append(agent_id)

		# self.sim.setAgentPosition(agent_id, pos)
		self.sim.setAgentPrefVelocity(agent_id, pref_vel)

	#ADD Obstacles (*** Assume all obstacles are made by 4 verticles, !!!! clockwise !!!!  ***)
	# in documentation it is counterclockwise, but I found clockwise works 
	#http://gamma.cs.unc.edu/RVO2/documentation/2.0/class_r_v_o_1_1_r_v_o_simulator.html#a0f4a896c78fc09240083faf2962a69f2

	def _init_add_obstacles(self, upper_left, upper_right, bottom_right, bottom_left):
		verticals = []
		vertical_id = self.sim.addObstacle([upper_left, upper_right, bottom_right, bottom_left])
		for j in range(4):
			verticals.append(vertical_id)
			vertical_id = self.sim.getNextObstacleVertexNo(vertical_id)
		self.world["obstacles_vertex_ids"].append(verticals)

	def update_pref_vel(self):
		for i in range(self.numAgents):
			pref_vel = self.comp_pred_vel(i)
			if i == 0:
				self.sim.setAgentPrefVelocity(self.world["agents_id"][i], pref_vel)
				# pass
			else:
				# self.sim.setAgentPrefVelocity(self.world["agents_id"][i], pref_vel)
				self.sim.setAgentPrefVelocity(self.world["agents_id"][i], (0,0))

	def comp_pred_vel(self, agent_id):
			pos = self.sim.getAgentPosition(self.world["agents_id"][agent_id])
			target_pos = self.world["targets_pos"][agent_id]
			angle = np.arctan2(target_pos[1]-pos[1],target_pos[0]-pos[0])
			pref_vel = (cos(angle), sin(angle))

			return pref_vel

	#part of the initial
	def _init_visworld(self):
		self.win = Tk()
		self.canvas = Canvas(self.win, width=self.pixelsize, height=self.pixelsize, background="#eaeaea")
		self.canvas.pack()

		self.visWorld = {}

		self.visWorld["bot_circles_id"] = []
		self.visWorld["vel_lines_id"] = []
		self.visWorld["pref_vel_lines_id"] = []
		self.visWorld["bot_laser_scan_lines_id"] = []
		#self.visWorld["bot_laser_scan_lines_id"].append([])

		# ADD Agents
		for i in range(len(self.world["agents_id"])):
			if i == 0:
				self.visWorld["bot_circles_id"].append(self.canvas.create_oval( \
					-self.radius, -self.radius, self.radius, self.radius, outline='', fill="#ff5733"))
			else:
				self.visWorld["bot_circles_id"].append(self.canvas.create_oval( \
					-self.radius, -self.radius, self.radius, self.radius, outline='', fill="#f8b739"))
			self.visWorld["vel_lines_id"].append(self.canvas.create_line(		\
					0, 0, self.radius, self.radius, arrow=LAST, width=2, fill="#f30067"))
			self.visWorld["pref_vel_lines_id"].append(self.canvas.create_line(	\
					0, 0, self.radius, self.radius, arrow=LAST, width=2, fill="#7bc67b"))

		# ADD Lasers
		# only one laser scan for now
		# for i in range(0, self.laser_num):
		# 	self.visWorld["bot_laser_scan_lines_id"][0].append(self.canvas.create_line(
		# 		0, 0, self.radius, self.radius, width=2, fill='purple'))

		for i in range(len(self.world["laserScans"])):
			laserScan = []
			for j in range(len(self.world["laserScans"][i])):
				laserScan.append(self.canvas.create_line(
					0, 0, self.radius, self.radius, width=2, fill='purple'))
			self.visWorld["bot_laser_scan_lines_id"].append(laserScan)

		#ADD Obstacles
		self.visWorld["obstacles_id"] = []
		for i in range(len(self.world["obstacles_vertex_ids"])):
			ids = self.world["obstacles_vertex_ids"][i]
			four_vertex_pos = [self.sim.getObstacleVertex(j) for j in ids]
			if i == 0:
				self.visWorld["obstacles_id"].append(self.canvas.create_polygon (\
					four_vertex_pos[0][0], four_vertex_pos[0][1],
					four_vertex_pos[1][0], four_vertex_pos[1][1],
					four_vertex_pos[2][0], four_vertex_pos[2][1],
					four_vertex_pos[3][0], four_vertex_pos[3][1],
					fill=""))
			else:
				self.visWorld["obstacles_id"].append(self.canvas.create_polygon(\
					four_vertex_pos[0][0], four_vertex_pos[0][1],
					four_vertex_pos[1][0], four_vertex_pos[1][1],
					four_vertex_pos[2][0], four_vertex_pos[2][1],
					four_vertex_pos[3][0], four_vertex_pos[3][1],
					fill="#444444"))

		#ADD targets
		self.visWorld["targets_id"] = []
		for i in range(len(self.world["targets_pos"])):
			self.visWorld["targets_id"].append(self.canvas.create_oval(	\
				0, 0, self.radius, self.radius, outline='', fill="#448ef6"))


	def _get_obs(self):
		for i in range(self.numAgents):
			agent_id = self.world["agents_id"][i]

			#Add pref_vel = norm(target_pos - current_pos) and current_vel into observation
			pref_vel = self.comp_pred_vel(agent_id)
			current_vel = self.sim.getAgentVelocity(agent_id)
			observation = [pref_vel[0],pref_vel[1],current_vel[0],current_vel[1]]

			#LASER
			relative_neighbor_lines = []
			neighbor_ids = []
			relative_obstacle_lines = []

			if self.sim.getAgentNumAgentNeighbors(agent_id) != 0:
				relative_neighbor_lines, neighbor_ids = self._obs_neighbor_agent_lines(agent_id)
			
			if self.sim.getAgentNumObstacleNeighbors(agent_id) != 0:
				relative_obstacle_lines = self._obs_obstacle_lines(agent_id)
			
			neighbor_vel = [self.sim.getAgentVelocity(id) for id in neighbor_ids]
			assert len(relative_neighbor_lines) == len(neighbor_vel)
			relative_neighbor_lines_with_vel = \
				[(relative_neighbor_lines[i],neighbor_vel[i]) for i in range(len(relative_neighbor_lines))]
			relative_obstacle_lines_with_vel = \
				[(relative_obstacle_lines[i],(0,0)) for i in range(len(relative_obstacle_lines))]

			lines_with_vel = relative_neighbor_lines_with_vel + relative_obstacle_lines_with_vel

			if len(lines_with_vel) > 0:
				laser_result = comp_laser(self.ray_lines, lines_with_vel)
			else:
				laser_result = [((0, 0), (0, 0))] * self.laser_num
			# print('------------------------laser_result\n',laser_result)

			for pos_vel in laser_result:
				observation += [pos_vel[0][0],pos_vel[0][1],pos_vel[1][0],pos_vel[1][1]]
			assert len(observation) == 4 + self.laser_num*4

			self.gym_obs['agent_'+str(i)] = observation
		# return np.array(observation, dtype=np.float32)
		return self.gym_obs

	def _obs_neighbor_agent_lines(self, agent_id):
		neighbor_lines = []
		neighbor_ids = []

		for i in range(self.sim.getAgentNumAgentNeighbors(agent_id)):
			# print("found agents",self.sim.getAgentAgentNeighbor(agent_id,i))
			neighbor_id = self.sim.getAgentAgentNeighbor(agent_id,i)
			neighbor_ids += [neighbor_id] * 8

			my_pos = self.sim.getAgentPosition(agent_id)
			neighbor_pos = self.sim.getAgentPosition(neighbor_id)
			relative_pos = (neighbor_pos[0]-my_pos[0],neighbor_pos[1]-my_pos[1])

			for line in self.approx_lines:
				neighbor_lines.append(((line[0][0] + relative_pos[0],line[0][1] + relative_pos[1]),
									   (line[1][0] + relative_pos[0],line[1][1] + relative_pos[1])))

		# print('neighbor_lines',neighbor_lines)
		#[((1.454, -0.054), (1.308, -0.408))...]
		return neighbor_lines, neighbor_ids

	#computer relative obstacle lines from one agent
	def _obs_obstacle_lines(self, agent_id):
		obstacle_lines = []
		# print('NumObstacleNeighbors',self.sim.getAgentNumObstacleNeighbors(agent_id))

		for i in range(self.sim.getAgentNumObstacleNeighbors(agent_id)):
			v1_id = self.sim.getAgentObstacleNeighbor(agent_id,i)
			v2_id = self.sim.getNextObstacleVertexNo(v1_id)
			# print("found wall",self.sim.getAgentObstacleNeighbor(agent_id,i))
			# print(self.sim.getObstacleVertex(v1_id),self.sim.getObstacleVertex(v2_id))
			v1_pos = self.sim.getObstacleVertex(v1_id)
			v2_pos = self.sim.getObstacleVertex(v2_id)
			agent_pos = self.sim.getAgentPosition(agent_id)

			obstacle_lines.append(((v1_pos[0] - agent_pos[0], v1_pos[1] - agent_pos[1]),
								   (v2_pos[0] - agent_pos[0], v2_pos[1] - agent_pos[1])))
		# print(obstacle_lines)
		#should be like [[(-1.95, -0.18), (-2.45, -0.18)]...]
		return obstacle_lines

	#computer laser lines from one agent
	def _init_comp_laser_rays(self):
		ray_lines = []
		d_theta = 2*pi / self.laser_num

		for i in range(self.laser_num):
			theta = i * d_theta
			laser_end_pos = (self.neighborDist*cos(theta), - self.neighborDist*sin(theta))
			ray_lines.append(((0,0), (laser_end_pos[0], laser_end_pos[1])))
		# print('ray_lines',ray_lines)
		# should be like [[(0, 0), (1.5, -0.0)], [(0, 0), (1.38, -0.57)]...]
		self.ray_lines = ray_lines

	#use polygon to approxmate circles
	def _init_comp_circle_approx(self):
		approx_lines = []
		d_theta = 2*pi / self.circle_approx_num
		fisrt_approx_pos = (self.radius*cos(0), - self.radius*sin(0))
		approx_pos = fisrt_approx_pos

		for i in range(1, self.circle_approx_num):
			theta = i * d_theta
			new_approx_pos = (self.radius*cos(theta), - self.radius*sin(theta))
			approx_lines.append(((approx_pos[0],approx_pos[1]), (new_approx_pos[0], new_approx_pos[1])))
			approx_pos = new_approx_pos

		approx_lines.append(((approx_pos[0],approx_pos[1]), (fisrt_approx_pos[0], fisrt_approx_pos[1])))
		# print('approx',approx)
		# should be like [[(1.5, -0.0), (1.06, -1.06)]...]
		self.approx_lines = approx_lines

	def done_test(self):
		for i in range(self.numAgents):
			agent_id = self.world["agents_id"][i]
			if self.agents_done[i] == 0:
				pos = self.sim.getAgentPosition(self.world["agents_id"][agent_id])
				# target_pos = self.world["targets_pos"][agent_id]
				# target_dist = (target_pos[1]-pos[1])**2 + (target_pos[0]-pos[0])**2
				if pos[0] < 2.0:
					self.agents_done[agent_id] = 1
					self.world["targets_pos"][agent_id] = (-10.0,5.0)
		if 0 not in self.agents_done:
			return True
		else:
			return False

	def step(self, action):

		for i in range(self.numAgents):
			agent_id = self.world["agents_id"][i]
			rl_vel = action['agent_'+str(i)]

			# normalize to max vel
			rl_vel /= np.linalg.norm(rl_vel)

			pref_vel = np.array(self.comp_pred_vel(agent_id))

			self.sim.setAgentPrefVelocity(agent_id, (float(rl_vel[0]),float(rl_vel[1])))
			self.sim.doStep()
			# self.update_pref_vel()
			if FLAG_DRAW:
				self.draw_update()

			orca_vel = self.sim.getAgentVelocity(agent_id)

			scale = 0.5
			R_goal = np.dot(orca_vel, pref_vel)
			# R_goal = np.dot(rl_vel, pref_vel)
			R_polite = np.dot(orca_vel, rl_vel)

			self.gym_rewards['agent_'+str(i)] = scale * R_goal + (1 - scale) * R_polite

		self.gym_dones['__all__'] = self.done_test()

		# reward += np.max([1/target_dist, 5])

		self.step_count += 1
		if self.step_count >= self.max_step:
			self.gym_dones['__all__'] = True
		self.gym_obs = self._get_obs()

		return self.gym_obs, self.gym_rewards, self.gym_dones, self.gym_infos

	#not used in RL, just for orca visualization
	def orca_step(self, action):
		self.sim.doStep()
		self.update_pref_vel()
		self.gym_obs = self._get_obs()

		# need to update for multiple robots
		# need to move this into render() or step()
		#self.world["laserScans"][0] = self.gym_obs[4:]
		
		self.draw_update()


	def reset(self):
		#clear gym parameters
		self.gym_obs = {}
		self.gym_rewards = {}
		self.gym_dones = {'__all__':False}
		self.gym_infos = {}

		#according to https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py
		#each agent needs a string name
		for i in range(self.numAgents):
			self.gym_obs['agent_'+str(i)] = [0.0] * (4 + self.laser_num * 4)
			self.gym_rewards['agent_'+str(i)] = 0
			self.gym_dones['agent_'+str(i)] = False
			self.gym_infos['agent_'+str(i)] = {}

		for i in range(self.numAgents):
			agent_id = self.world["agents_id"][i]
			pos = (uniform(self.envsize * 0.5, self.envsize), uniform(0, self.envsize))
			self.sim.setAgentPosition(agent_id, pos)

		self.update_pref_vel()
		self.step_count = 0
		self.agents_done = [0] * self.numAgents

		return self._get_obs()

	#gym native render, should be useless
	def render(self, mode='human'):
		self.draw_update()

	def seed(self, seed=None):
		self.rng, seed = seeding.np_random(seed)
		return [seed]

	#gym native func, should never be called
	def close(self):
		print("CALLING close()")
		input()


	def draw_world(self):
		scale = self.pixelsize / self.envsize

		#draw agents
		for i in range(len(self.world["agents_id"])):
			self.canvas.coords(self.visWorld["bot_circles_id"][i],
						scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[0] - self.radius),
						scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[1] - self.radius),
						scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[0] + self.radius),
						scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[1] + self.radius))
			self.canvas.coords(self.visWorld["vel_lines_id"][i],
						scale * self.sim.getAgentPosition(self.world["agents_id"][i])[0],
						scale * self.sim.getAgentPosition(self.world["agents_id"][i])[1],
						scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[0] + 2 * self.radius *
								self.sim.getAgentVelocity(self.world["agents_id"][i])[0]),
						scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[1] + 2 * self.radius *
								self.sim.getAgentVelocity(self.world["agents_id"][i])[1]))
			self.canvas.coords(self.visWorld["pref_vel_lines_id"][i],
						scale * self.sim.getAgentPosition(self.world["agents_id"][i])[0],
						scale * self.sim.getAgentPosition(self.world["agents_id"][i])[1],
						scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[0] + 2 * self.radius *
								self.sim.getAgentPrefVelocity(self.world["agents_id"][i])[0]),
						scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[1] + 2 * self.radius *
								self.sim.getAgentPrefVelocity(self.world["agents_id"][i])[1]))

		#draw obstacles
		for i in range(len(self.world["obstacles_vertex_ids"])):
			ids = self.world["obstacles_vertex_ids"][i]
			self.canvas.coords(self.visWorld["obstacles_id"][i],
						scale * self.sim.getObstacleVertex(ids[0])[0],
						scale * self.sim.getObstacleVertex(ids[0])[1],
						scale * self.sim.getObstacleVertex(ids[1])[0],
						scale * self.sim.getObstacleVertex(ids[1])[1],
						scale * self.sim.getObstacleVertex(ids[2])[0],
						scale * self.sim.getObstacleVertex(ids[2])[1],
						scale * self.sim.getObstacleVertex(ids[3])[0],
						scale * self.sim.getObstacleVertex(ids[3])[1])

		#draw targets
		for i in range(len(self.world["targets_pos"])):
			self.canvas.coords(self.visWorld["targets_id"][i],
						scale * (self.world["targets_pos"][i][0] - self.radius),
						scale * (self.world["targets_pos"][i][1] - self.radius),
						scale * (self.world["targets_pos"][i][0] + self.radius),
						scale * (self.world["targets_pos"][i][1] + self.radius))

		'''
		#draw Lasers
		for i in range(1):  # for i in range(len(self.world["laserScans"])):
			vis_i = 0
			for j in range(0, len(self.world["laserScans"][i]), 4):
				pos_vel = self.world["laserScans"][i][j:j + 4]
				self.canvas.coords(self.visWorld["bot_laser_scan_lines_id"][i][vis_i],
								scale * self.sim.getAgentPosition(self.world["agents_id"][i])[0],
								scale * self.sim.getAgentPosition(self.world["agents_id"][i])[1],
								scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[0] + pos_vel[0]),
								scale * (self.sim.getAgentPosition(self.world["agents_id"][i])[1] + pos_vel[1]))
				vis_i += 1
		'''


	def draw_update(self):
		self.draw_world()
		self.win.update_idletasks()
		self.win.update()
		time.sleep(self.timeStep)


if __name__ == "__main__":
	CA = Collision_Avoidance_Env()
	for i in range(2000):
		CA.orca_step((0,0))