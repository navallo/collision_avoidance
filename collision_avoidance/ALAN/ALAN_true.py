from math import sin, cos, sqrt, exp, pi
import time
from random import uniform

import gym
# from gym import error, spaces, utils
from gym.utils import seeding

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import numpy as np
FLAG_DRAW = True
# FLAG_DRAW = False
if FLAG_DRAW:
    from tkinter import *
import rvo2

from collision_avoidance.envs.utils import *

class Collision_Avoidance_Sim():

    def __init__(self, numAgents=50, scenario="crowd", visualize=True):

        # ORCA config
        self.timeStep = 1/60.
        self.neighborDist = 5#1.5
        self.maxNeighbors = 10#5
        self.timeHorizon = 1.5
        self.radius = 0.5
        self.maxSpeed = 1

        self.sim = rvo2.PyRVOSimulator(timeStep=self.timeStep,
                                       neighborDist=self.neighborDist,
                                       maxNeighbors=self.maxNeighbors,
                                       timeHorizon=self.timeHorizon,
                                       timeHorizonObst=self.timeHorizon,
                                       radius=self.radius,
                                       maxSpeed=self.maxSpeed)

        # ALAN config
        self.online_actions = [(1, 0),
            (0.70711, 0.70711),
            (0, 1),
            (-0.70711, 0.70711),
            (-1, 0),
            (-0.70711, -0.70711),
            (0, -1),
            (0.70711, -0.70711)]
        self.timewindow = 2
        self.online_temp = 0.05

        # world config
        self.numAgents = numAgents
        self.scenario = scenario

        self.pixelsize = 1000
        self.envsize = self.radius*numAgents

        self.step_count = 0
        self.max_step = int((10/self.timeStep) * self.numAgents)
        print(self.max_step)

        self.agents_done = [0] * self.numAgents

        self._init_world()
        self.visualize = visualize
        self.play_speed = 4
        if visualize:
            self._init_visworld()

        # start timer
        self.t_start = time.time()

    def reset(self, numAgents=50, scenario="crowd"):
        # ORCA config
        self.sim = rvo2.PyRVOSimulator(timeStep=self.timeStep,
                                       neighborDist=self.neighborDist,
                                       maxNeighbors=self.maxNeighbors,
                                       timeHorizon=self.timeHorizon,
                                       timeHorizonObst=self.timeHorizon,
                                       radius=self.radius,
                                       maxSpeed=self.maxSpeed)


        # world config
        self.step_count = 0
        self.max_step = 2000

        self.agents_done = [0] * self.numAgents

        self._init_world()
        self.play_speed = 4
        if self.visualize:
            self.win.destroy()
            self._init_visworld()

    def run_sim(self, mode=1):
        # check if valid mode is given
        if mode != 1 & mode != 0:
            mode = 1

        # step through simulation
        self.t_start = time.time()
        success = False
        for i in range(self.max_step):
            if mode == 1:
                self.online_step()
            else:
                self.orca_step()
            success = self.done_test()
            if success:
                break

        return self.step_count*self.timeStep, success

    def _init_world(self):
        self.world = {}
        self.world["agents_id"] = []
        self.world["obstacles_vertex_ids"] = []
        self.world["laserScans"] = []
        self.world["action_weights"] = []
        self.world["action_times"] = []
        self.world["targets_pos"] = []

        if self.scenario == "congested":
            self._init_world_congested()
        elif self.scenario == "deadlock":
            self._init_world_deadlock()
        elif self.scenario == "blocks":
            self._init_world_blocks()
        elif self.scenario == "circle":
            self._init_world_circle()
        elif self.scenario == "crowd":
            self._init_world_crowd()
        elif self.scenario == "incoming":
            self._init_world_incoming()
        else:
            print(self.scenario, "is not a valid scenario")

    def _init_world_congested(self):
        # adjust environment size
        self.envsize = 1.5 * self.radius * self.numAgents
        # Add agents
        for i in range(self.numAgents):
            pos = (uniform(self.envsize * 0.2, self.envsize), uniform(0, self.envsize))
            angle = uniform(0, 2 * pi)
            pref_vel = (cos(angle), sin(angle))
            self._init_add_agents(pos, pref_vel)

            # set target
            target_pos = ((0.1 * self.envsize - 1.0, self.envsize / 2),
                          (0.1 * self.envsize - self.envsize, self.envsize / 2))
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

        self.update_pref_vel()

        # border obstacle
        self._init_add_obstacles((-self.envsize, 0.0), (-self.envsize, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        # internal obstacles
        self._init_add_obstacles((0.1 * self.envsize, 0.0),
                                 (0.1 * self.envsize + 0.5, 0.0),
                                 (0.1 * self.envsize + 0.5, self.envsize / 2 - 1.25 * self.radius),
                                 (0.1 * self.envsize, self.envsize / 2 - 1.25 * self.radius))

        self._init_add_obstacles((0.1 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (0.1 * self.envsize + 0.5, self.envsize / 2 + 1.25 * self.radius),
                                 (0.1 * self.envsize + 0.5, self.envsize),
                                 (0.1 * self.envsize, self.envsize))

        self.sim.processObstacles()

    def _init_world_incoming(self):
        # adjust environment size
        self.envsize = (1.5 * self.radius * self.numAgents) * 2

        # Add single agent
        pos = (0.1 * self.envsize, self.envsize / 2)
        angle = uniform(0, 2 * pi)
        pref_vel = (cos(angle), sin(angle))
        self._init_add_agents(pos, pref_vel)

        # set target
        target_pos = ((0.9 * self.envsize, self.envsize / 2),
                      (0.9 * self.envsize, self.envsize / 2))
        self.world["targets_pos"].append(target_pos)

        self.world["action_weights"].append([0.0] * len(self.online_actions))
        self.world["action_times"].append([0.0] * len(self.online_actions))

        # Add agents
        numAgents_incoming = self.numAgents - 1
        agent_block_len = sqrt(numAgents_incoming)
        x_inc = 3*self.radius
        y_inc = 2.1*self.radius
        y_start = self.envsize / 2 - ((y_inc*agent_block_len) / 2)

        x_pos = 0.8*self.envsize
        y_pos = y_start
        for i in range(numAgents_incoming):
            pos = (x_pos, y_pos)
            angle = uniform(0, 2 * pi)
            pref_vel = (cos(angle), sin(angle))
            self._init_add_agents(pos, pref_vel)

            # set target
            target_pos = ((x_pos - 0.7*self.envsize, y_pos),
                          (x_pos - 0.7*self.envsize, y_pos))
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

            # increment position
            y_pos += y_inc
            if(y_pos > y_start + y_inc*agent_block_len):
                x_pos += x_inc
                y_pos = y_start


        self.update_pref_vel()

        # border obstacle
        self._init_add_obstacles((0.0, 0.0), (0.0, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        self.sim.processObstacles()

    def _init_world_crowd(self):
        # adjust environment size
        self.envsize = (self.radius * self.numAgents) / 2
        # Add agents
        for i in range(self.numAgents):
            pos = (uniform(0, self.envsize), uniform(0, self.envsize))
            angle = uniform(0, 2 * pi)
            pref_vel = (cos(angle), sin(angle))
            self._init_add_agents(pos, pref_vel)

            # set target
            target = (uniform(0, self.envsize), uniform(0, self.envsize))
            target_pos = (target, target)
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

        self.update_pref_vel()

        # border obstacle
        self._init_add_obstacles((0.0, 0.0), (0.0, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        self.sim.processObstacles()

    def _init_world_circle(self):
        # adjust environment size
        circle_circumference = self.radius*3*self.numAgents
        circle_radius = circle_circumference/(2*pi)
        self.envsize = 2*circle_radius + 4*self.radius

        theta_inc = (2*pi)/self.numAgents
        theta = 0
        # Add agents
        for i in range(self.numAgents):
            pos = (self.envsize/2 + circle_radius*cos(theta),
                   self.envsize/2 + circle_radius*sin(theta))
            angle = uniform(0, 2 * pi)
            pref_vel = (cos(angle), sin(angle))
            self._init_add_agents(pos, pref_vel)

            # set target
            target = (self.envsize/2 + circle_radius*cos(theta + pi),
                      self.envsize/2 + circle_radius*sin(theta + pi))
            target_pos = (target, target)
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

            theta += theta_inc

        self.update_pref_vel()

        # border obstacle
        self._init_add_obstacles((0.0, 0.0), (0.0, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        self.sim.processObstacles()

    def _init_world_blocks(self):
        # adjust environment size
        self.envsize = 3 * self.radius * self.numAgents

        x_pos = 1.5 * self.radius
        y_pos = 1.5 * self.radius
        y_inc = 3 * self.radius
        # Add agents
        for i in range(self.numAgents):
            pos = (x_pos, y_pos)
            angle = uniform(0, 2 * pi)
            pref_vel = (cos(angle), sin(angle))
            self._init_add_agents(pos, pref_vel)

            # set target
            target = (self.envsize - 1.5 * self.radius, y_pos)
            target_pos = (target, target)
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

            y_pos += y_inc

        self.update_pref_vel()

        # border obstacle
        self._init_add_obstacles((0.0, 0.0), (0.0, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        # blocks
        num_blocks = 4
        block_size = self.envsize/(num_blocks*2)
        for i in range(num_blocks):
            pos = (uniform(block_size, self.envsize - block_size),
                   uniform(0, self.envsize))
            self._init_add_obstacles((pos[0] - block_size / 2, pos[1] - block_size / 2),
                                     (pos[0] + block_size / 2, pos[1] - block_size / 2),
                                     (pos[0] + block_size / 2, pos[1] + block_size / 2),
                                     (pos[0] - block_size / 2, pos[1] + block_size / 2))

        self.sim.processObstacles()

    def _init_world_deadlock(self):
        # adjust environment size
        self.envsize = (self.radius * self.numAgents)*5
        # Add agents
        pos_y = self.envsize / 2
        pos_x = 0.2 * self.envsize
        x_inc = -3 * self.radius
        for i in range(0, int(self.numAgents / 2)):
            pos = (pos_x, pos_y)
            angle = uniform(0, 2 * pi)
            pref_vel = (cos(angle), sin(angle))
            self._init_add_agents(pos, pref_vel)

            # set target
            target_pos = ((0.9 * self.envsize, self.envsize / 2),
                          (0.9 * self.envsize + self.envsize, self.envsize / 2))
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))
            pos_x += x_inc

        pos_x = 0.8 * self.envsize
        x_inc = 3 * self.radius
        for i in range(int(self.numAgents / 2), self.numAgents):
            pos = (pos_x, pos_y)
            angle = uniform(0, 2 * pi)
            pref_vel = (cos(angle), sin(angle))
            self._init_add_agents(pos, pref_vel)

            # set target
            target_pos = ((0.1 * self.envsize, self.envsize / 2),
                          (0.1 * self.envsize - self.envsize, self.envsize / 2))
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))
            pos_x += x_inc

        self.update_pref_vel()

        # border obstacle
        self._init_add_obstacles((-self.envsize, 0.0), (-self.envsize, self.envsize),
                                 (2 * self.envsize, self.envsize), (2 * self.envsize, 0.0))

        # internal obstacles

        # left funnel
        self._init_add_obstacles((0.0, 0.0),
                                 (0.0 + 0.5, 0.0),
                                 (0.2 * self.envsize + 0.5, self.envsize / 2 - 1.25 * self.radius),
                                 (0.2 * self.envsize, self.envsize / 2 - 1.25 * self.radius))

        self._init_add_obstacles((0.0, self.envsize),
                                 (0.2 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (0.2 * self.envsize + 0.5, self.envsize / 2 + 1.25 * self.radius),
                                 (0.0 + 0.5, self.envsize))

        # right funnel
        self._init_add_obstacles((self.envsize - 0.5, 0.0),
                                 (self.envsize, 0.0),
                                 (0.8 * self.envsize, self.envsize / 2 - 1.25 * self.radius),
                                 (0.8 * self.envsize - 0.5, self.envsize / 2 - 1.25 * self.radius))

        self._init_add_obstacles((self.envsize - 0.5, self.envsize),
                                 (0.8 * self.envsize - 0.5, self.envsize / 2 + 1.25 * self.radius),
                                 (0.8 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (self.envsize, self.envsize))

        # tube
        self._init_add_obstacles((0.2 * self.envsize, (self.envsize / 2 - 1.25 * self.radius) - 0.5),
                                 (0.8 * self.envsize, (self.envsize / 2 - 1.25 * self.radius) - 0.5),
                                 (0.8 * self.envsize, self.envsize / 2 - 1.25 * self.radius),
                                 (0.2 * self.envsize, self.envsize / 2 - 1.25 * self.radius))

        self._init_add_obstacles((0.2 * self.envsize, (self.envsize / 2 + 1.25 * self.radius) + 0.5),
                                 (0.2 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (0.8 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (0.8 * self.envsize, (self.envsize / 2 + 1.25 * self.radius) + 0.5))

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
            pref_vel = self.comp_pref_vel(i)
            self.sim.setAgentPrefVelocity(self.world["agents_id"][i], pref_vel)

    def comp_pref_vel(self, agent_id):
        pos = self.sim.getAgentPosition(self.world["agents_id"][agent_id])
        target_pos = self.world["targets_pos"][agent_id][0]
        angle = np.arctan2(target_pos[1]-pos[1],target_pos[0]-pos[0])
        pref_vel = (cos(angle), sin(angle))

        return pref_vel

    #part of the initial
    def _init_visworld(self):
        self.win = Tk()
        self.win.title(self.scenario + ": " + str(self.numAgents))
        self.canvas = Canvas(self.win, width=self.pixelsize, height=self.pixelsize, background="#eaeaea")
        self.canvas.pack()

        self.visWorld = {}

        self.visWorld["bot_circles_id"] = []
        self.visWorld["vel_lines_id"] = []
        self.visWorld["pref_vel_lines_id"] = []

        # ADD targets
        self.visWorld["targets_id"] = []
        for i in range(len(self.world["targets_pos"])):
            self.visWorld["targets_id"].append(self.canvas.create_oval(
                0, 0, self.radius, self.radius, outline='', fill="blue"))

        # ADD Agents
        for i in range(len(self.world["agents_id"])):
            self.visWorld["bot_circles_id"].append(self.canvas.create_oval(
                -self.radius, -self.radius, self.radius, self.radius, outline='', fill="yellow"))
            self.visWorld["vel_lines_id"].append(self.canvas.create_line(
                    0, 0, self.radius, self.radius, arrow=LAST, width=2, fill="red"))
            self.visWorld["pref_vel_lines_id"].append(self.canvas.create_line(
                    0, 0, self.radius, self.radius, arrow=LAST, width=2, fill="green"))

        #ADD Obstacles
        self.visWorld["obstacles_id"] = []
        for i in range(len(self.world["obstacles_vertex_ids"])):
            ids = self.world["obstacles_vertex_ids"][i]
            four_vertex_pos = [self.sim.getObstacleVertex(j) for j in ids]
            if i == 0:
                self.visWorld["obstacles_id"].append(self.canvas.create_polygon (
                    four_vertex_pos[0][0], four_vertex_pos[0][1],
                    four_vertex_pos[1][0], four_vertex_pos[1][1],
                    four_vertex_pos[2][0], four_vertex_pos[2][1],
                    four_vertex_pos[3][0], four_vertex_pos[3][1],
                    fill=""))
            else:
                self.visWorld["obstacles_id"].append(self.canvas.create_polygon(
                    four_vertex_pos[0][0], four_vertex_pos[0][1],
                    four_vertex_pos[1][0], four_vertex_pos[1][1],
                    four_vertex_pos[2][0], four_vertex_pos[2][1],
                    four_vertex_pos[3][0], four_vertex_pos[3][1],
                    fill="black"))

    def done_test(self):
        for i in range(self.numAgents):
            agent_id = self.world["agents_id"][i]
            if self.agents_done[i] == 0:
                # check if agent has reached goal
                pos = self.sim.getAgentPosition(self.world["agents_id"][agent_id])
                t_pos = self.world["targets_pos"][agent_id][0]
                if sqrt((pos[0] - t_pos[0])**2 + (pos[1] - t_pos[1])**2) < 2 * self.radius:
                    self.agents_done[agent_id] = 1
                    self.world["targets_pos"][agent_id] = (self.world["targets_pos"][agent_id][1],
                                                           self.world["targets_pos"][agent_id][1])
        if 0 not in self.agents_done:
            return True
        else:
            return False

    def online_step(self):

        action_vels = []
        pref_vels = []
        action_ids = []
        for i in range(self.numAgents):
            agent_id = self.world["agents_id"][i]

            weights = np.array([i for i in self.world["action_weights"][agent_id]])
            ps = np.exp(weights / self.online_temp)
            ps /= np.sum(ps)

            action_id = int(np.random.choice(len(self.online_actions), 1, p=ps))
            action = self.online_actions[action_id]
            action_ids.append(action_id)

            pref_vel = np.array(self.comp_pref_vel(agent_id))
            pref_vels.append(pref_vel)

            goal_theta = np.arctan2(pref_vel[1], pref_vel[0])
            theta = np.arctan2(action[1], action[0])
            goal_theta += theta
            action_vel = (cos(goal_theta), sin(goal_theta))
            action_vels.append(action_vel)

            self.sim.setAgentPrefVelocity(agent_id, (float(action_vel[0]),float(action_vel[1])))

        self.sim.doStep()
        if self.visualize:
            self.draw_update()

        for i in range(self.numAgents):
            pref_vel = pref_vels[i]
            action_vel = action_vels[i]
            action_id = action_ids[i]

            agent_id = self.world["agents_id"][i]

            orca_vel = self.sim.getAgentVelocity(agent_id)

            # calculate reward
            scale = 0.7
            R_goal = np.dot(orca_vel, pref_vel)
            R_polite = np.dot(orca_vel, action_vel)
            R = scale*R_goal + (1-scale)*R_polite

            # increment time window of action weights
            for act_id in range(len(self.online_actions)):
                self.world["action_times"][agent_id][act_id] += self.timeStep
                if self.world["action_times"][agent_id][act_id] >= self.timewindow:
                    self.world["action_times"][agent_id][act_id] = 0
                    self.world["action_weights"][agent_id][act_id] = 0

            # update weight
            self.world["action_weights"][agent_id][action_id] = R

        # increment steps
        self.step_count += 1


    #not used in RL, just for orca visualization
    def orca_step(self):
        self.sim.doStep()
        self.update_pref_vel()

        if FLAG_DRAW:
            self.draw_update()

        self.step_count += 1

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

        # draw targets
        for i in range(len(self.world["targets_pos"])):
            self.canvas.coords(self.visWorld["targets_id"][i],
                               scale * (self.world["targets_pos"][i][0][0] - self.radius),
                               scale * (self.world["targets_pos"][i][0][1] - self.radius),
                               scale * (self.world["targets_pos"][i][0][0] + self.radius),
                               scale * (self.world["targets_pos"][i][0][1] + self.radius))

        # draw agents
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
            # update color of agent
            if self.agents_done[i] == 1:
                self.canvas.itemconfig(self.visWorld["bot_circles_id"][i], fill="purple")

        # draw obstacles
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

    def draw_update(self):
        # draw world and update window
        self.draw_world()
        self.win.update_idletasks()
        self.win.update()
        # calculate display delay
        desired_time = self.step_count*self.timeStep
        sleep_time = desired_time/self.play_speed - (time.time() - self.t_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    # congested
    # crowd
    # deadlock
    # circle
    # blocks
    CA = Collision_Avoidance_Sim(21, "incoming", True)
    print(CA.run_sim(0))
    # CA.reset()
    # print(CA.run_sim(1))