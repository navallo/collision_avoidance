from math import sin, cos, sqrt, exp
import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from tkinter import *
import rvo2


class Collision_Avoidance_Env(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.timeStep = 1/120.
		self.neighborDist = 1.5
		self.maxNeighbors = 5
		self.timeHorizon = 1.5
		self.radius = 0.5  # 2
		self.maxSpeed = 1
		self.velocity = 2#(1, 1)

		self.numAgents = 10

		self.pixelsize = 500
		self.framedelay = 30
		self.envsize = 10

		self.rng, seed = seeding.np_random()

		self.sim = rvo2.PyRVOSimulator(self.timeStep,
									   self.neighborDist, 
									   self.maxNeighbors, 
									   self.timeHorizon, 
									   self.radius, 
									   self.maxSpeed, 
									   self.velocity)
		self._init_world()
		self._init_visworld()
		self.draw_update()
		
		self.reset()
	
	#part of the initial
	def _init_world(self):
		self.world = {}
		self.world["agents_id"] = []
		self.world["obstacles_vertex_ids"] = []

		self.world["targets_pos"] = []

		#Add agents
		for i in range(self.numAgents):
			pos = (self.rng.uniform(self.envsize * 0.5, self.envsize), self.rng.uniform(0, self.envsize))
			angle = self.rng.uniform(0, 2 * np.pi)
			pref_vel = (cos(angle), sin(angle))
			self._init_add_agents(pos, pref_vel)

			#set target
			target_pos = (1.0,5.0)
			self.world["targets_pos"].append(target_pos)

		self.update_pref_vel()

		#Add an obstacle
		self._init_add_obstacles((2.0,0.0),(2.5,0.0),(2.5,4.2),(2.0,4.2))
		self._init_add_obstacles((2.0,5.8),(2.5,5.8),(2.5,10.0),(2.0,10.0))
		self.sim.processObstacles()

	def _init_add_agents(self, pos, pref_vel):
		agents_id = self.sim.addAgent(pos, 
									  self.neighborDist, 
									  self.maxNeighbors, 
									  self.timeHorizon, 
									  1.5, 
									  self.radius, 
									  self.maxSpeed, 
									  pref_vel)
		self.world["agents_id"].append(agents_id)

		# self.sim.setAgentPosition(agents_id, pos)
		self.sim.setAgentPrefVelocity(agents_id, pref_vel)

	#ADD Obstacles (*** Assume all obstacles are made by 4 verticles, upper left, upper right, bottom right, bottom left ***)
	def _init_add_obstacles(self, upper_left, upper_right, bottom_right, bottom_left):
		verticals = []
		vertical_id = self.sim.addObstacle([upper_left, upper_right, bottom_right, bottom_left])
		for j in range(4):
			verticals.append(vertical_id)
			vertical_id = self.sim.getNextObstacleVertexNo(vertical_id)
		self.world["obstacles_vertex_ids"].append(verticals)

	def update_pref_vel(self):
		for i in range(self.numAgents):
			pos = self.sim.getAgentPosition(self.world["agents_id"][i])
			target_pos = self.world["targets_pos"][i]
			angle = np.arctan2(target_pos[1]-pos[1],target_pos[0]-pos[0]) # + np.pi ???
			pref_vel = (cos(angle), sin(angle))
			self.sim.setAgentPrefVelocity(self.world["agents_id"][i], pref_vel)


	#part of the initial
	def _init_visworld(self):
		self.win = Tk()
		self.canvas = Canvas(self.win, width=self.pixelsize, height=self.pixelsize, background="#eaeaea")
		self.canvas.pack()

		self.visWorld = {}

		#ADD Agents
		self.visWorld["bot_circles_id"] = []
		self.visWorld["vel_lines_id"] = []
		self.visWorld["pref_vel_lines_id"] = []

		for i in range(len(self.world["agents_id"])):
			self.visWorld["bot_circles_id"].append(self.canvas.create_oval(0, 0, self.radius, self.radius, outline='', fill="#f8b739"))
			self.visWorld["vel_lines_id"].append(self.canvas.create_line(0, 0, self.radius, self.radius, arrow=LAST, width=2, fill="#f30067"))
			self.visWorld["pref_vel_lines_id"].append(self.canvas.create_line(0, 0, self.radius, self.radius, arrow=LAST, width=2, fill="#7bc67b"))

		#ADD Obstacles
		self.visWorld["obstacles_id"] = []
		for i in range(len(self.world["obstacles_vertex_ids"])):
			ids = self.world["obstacles_vertex_ids"][i]
			four_vertex_pos = [self.sim.getObstacleVertex(j) for j in ids]
			self.visWorld["obstacles_id"].append(self.canvas.create_polygon(four_vertex_pos[0][0], four_vertex_pos[0][1],
																			four_vertex_pos[1][0], four_vertex_pos[1][1],
																			four_vertex_pos[2][0], four_vertex_pos[2][1],
																			four_vertex_pos[3][0], four_vertex_pos[3][1],
																			fill="#444444"))

		#ADD targets
		self.visWorld["targets_id"] = []
		for i in range(len(self.world["targets_pos"])):
			self.visWorld["targets_id"].append(self.canvas.create_oval(0, 0, self.radius, self.radius, outline='', fill="#c40b13"))



	def _get_obs(self):
		pass

	def step(self, action):
		self.sim.doStep()
		self.update_pref_vel()
		self.draw_update()
		# return ob, reward, done, {}

	def reset(self):
		pass
		# return self.env._get_obs()

	#gym native render, should be useless
	def render(self, mode='human'):
		pass

	#gym native func, should never be called
	def close(self):
		print("CALLING close()")
		input()


	def draw_world(self):
		scale = self.pixelsize / self.envsize

		#draw targets
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

	def draw_update(self):
		self.draw_world()
		self.win.update_idletasks()
		self.win.update()
		time.sleep(self.timeStep)


if __name__ == "__main__":
	CA = Collision_Avoidance_Env()
	for i in range(2000):
		CA.step(1)