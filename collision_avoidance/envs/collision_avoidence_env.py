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

#this is a simplified theta version (laser 0 always points to target)
class Collision_Avoidance_Env(gym.Env, MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, numAgents = 10, scenario="congested"):
        self.timeStep = 1/10
        self.neighborDist = 5
        self.maxNeighbors = 10
        self.timeHorizon = 1.5
        self.radius = 0.5
        self.maxSpeed = 1
        self.laser_num = 16
        self.circle_approx_num = 8

        self.sim = rvo2.PyRVOSimulator(timeStep=self.timeStep,
                                       neighborDist=self.neighborDist, 
                                       maxNeighbors=self.maxNeighbors, 
                                       timeHorizon=self.timeHorizon, 
                                       timeHorizonObst=self.timeHorizon, 
                                       radius=self.radius, 
                                       maxSpeed=self.maxSpeed)

        self.numAgents = numAgents
        self.scenario = scenario

        self.pixelsize = 500 #1000
        self.envsize = self.radius*numAgents
        self.play_speed = 4

        self.step_count = 0
        self.max_step = int((10/self.timeStep) * self.numAgents)

        self.agents_done = [0] * self.numAgents

        #gym parmaters
        #Discrete action space
        self.action_size = 8
        self.action_space = gym.spaces.Discrete(self.action_size)

        #Continuous action space
        # self.action_space = gym.spaces.Box(low=-pi, high=pi, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-self.neighborDist, high=self.neighborDist, shape=(self.laser_num,))


        # self.seed()

        self._init_comp_laser_rays()
        self._init_comp_circle_approx()

        self._init_world()
        if FLAG_DRAW:
            self._init_visworld()
            # self.draw_update()

        self.reset()
        self.t_start = time.time()

    def run_sim(self, mode=1):
        # check if valid mode is given
        if mode != 1 & mode != 0:
            mode = 1

        # step through simulation
        self.t_start = time.time()
        success = False
        for i in range(self.max_step):
            if mode == 1:
                print('no online step')
                # self.online_step()
                return -1
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
        #self.world["laserScans"].append([])  # for now only one laser scan

        self.world["targets_pos"] = []

        if self.scenario == "congested":
            self._init_world_congested()
        elif self.scenario == "incoming":
            self._init_world_incoming()
        elif self.scenario == "crowd":
            self._init_world_crowd()
        elif self.scenario == "circle":
            self._init_world_circle()
        elif self.scenario == "blocks":
            self._init_world_blocks()
        elif self.scenario == "deadlock":
            self._init_world_deadlock()
        else:
            print(self.scenario, "is not a valid scenario")
            return -1
        #Add lasers
        for i in range(self.numAgents):
            self.world["laserScans"].append([0.0] * (self.laser_num * 4))

    def _init_world_congested(self):
        # adjust environment size
        self.envsize = sqrt(2 * self.radius * self.numAgents) * 3
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
        self.envsize = sqrt(2 * self.radius * self.numAgents) * 10

        # Add single agent
        pos = (0.1 * self.envsize, self.envsize / 2)
        angle = uniform(0, 2 * pi)
        pref_vel = (cos(angle), sin(angle))
        self._init_add_agents(pos, pref_vel)

        # set target
        target_pos = ((0.9 * self.envsize, self.envsize / 2),
                      (0.9 * self.envsize, self.envsize / 2))
        self.world["targets_pos"].append(target_pos)


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
        self.envsize = sqrt(2 * self.radius * self.numAgents) * 2
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
        self.envsize = sqrt(2 * self.radius * self.numAgents) * 10
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
        self.visWorld["bot_laser_scan_lines_id"] = []
        #self.visWorld["bot_laser_scan_lines_id"].append([])

        #ADD targets
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

        '''
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
        '''

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

    def _get_obs(self):
        for i in range(self.numAgents):
            agent_id = self.world["agents_id"][i]

            #Add pref_vel = norm(target_pos - current_pos) and current_vel into observation
            pref_vel = self.comp_pref_vel(agent_id)
            current_vel = self.sim.getAgentVelocity(agent_id)
            # observation = [pref_vel[0],pref_vel[1],current_vel[0],current_vel[1]]
            observation = []

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

            # convert to refrence frame of robot
            #agent_pos = self.sim.getAgentPosition(agent_id)

            if len(lines_with_vel) > 0:
                laser_result = comp_laser_with_rot(self.ray_lines, lines_with_vel, pref_vel)
            else:
                laser_result = [((0, 0), (0, 0))] * self.laser_num
            # print('------------------------laser_result\n',laser_result)

            for pos_vel in laser_result:
                # observation += [pos_vel[0][0],pos_vel[0][1],pos_vel[1][0],pos_vel[1][1]]
                observation += [sqrt(pos_vel[0][0]*pos_vel[0][0] + pos_vel[0][1]*pos_vel[0][1])]
            assert len(observation) == self.laser_num

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

    def step(self, action):

        rl_vels = []
        pref_vels = []
        for i in range(self.numAgents):
            agent_id = self.world["agents_id"][i]
            action_id = action['agent_'+str(i)]
            theta = (action_id/self.action_size)*2*pi

            pref_vel = np.array(self.comp_pref_vel(agent_id))
            pref_vels.append(pref_vel)
            
            goal_theta = np.arctan2(pref_vel[1], pref_vel[0])
            goal_theta += theta
            rl_vel = (cos(goal_theta), sin(goal_theta))
            rl_vels.append(rl_vel)

            self.sim.setAgentPrefVelocity(agent_id, (float(rl_vel[0]),float(rl_vel[1])))

        self.sim.doStep()
        if FLAG_DRAW:
            self.draw_update()

        for i in range(self.numAgents):
            pref_vel = pref_vels[i]
            rl_vel = rl_vels[i]

            agent_id = self.world["agents_id"][i]
            orca_vel = self.sim.getAgentVelocity(agent_id)

            scale = 0.5
            R_goal = np.dot(orca_vel, pref_vel)
            R_greedy = np.dot(rl_vel, pref_vel)
            R_polite = np.dot(orca_vel, rl_vel)
            # self.gym_rewards['agent_' + str(i)] = scale*R_goal + (1-scale)*R_polite
            self.gym_rewards['agent_' + str(i)] = 0.5*R_goal + 0.5*R_greedy + 2*R_polite
            self.gym_rewards['agent_' + str(i)] += -0.2 if self.agents_done[agent_id] == 0 else 0

        self.gym_obs = self._get_obs()

        self.gym_dones['__all__'] = self.done_test()
        self.step_count += 1
        if self.step_count >= self.max_step:
            self.gym_dones['__all__'] = True
        if self.gym_dones['__all__'] == True:
            print('episode_steps,', self.step_count)

        return self.gym_obs, self.gym_rewards, self.gym_dones, self.gym_infos

    def rotate_laser_scan(self, laser, pref_vel):
        theta = np.arctan2(pref_vel[1], pref_vel[0])

        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

        new_laser = []
        for i in range(0, len(laser), 4):
            point = laser[i:i+2]
            vel = laser[i+2:i+4]

            point = np.array(point)
            vel = np.array(vel) + point

            point = rot @ point
            vel = rot @ vel

            vel = vel - point

            new_laser.append(point[0])
            new_laser.append(point[1])
            new_laser.append(vel[0])
            new_laser.append(vel[1])

        return new_laser



    #not used in RL, just for orca visualization
    def orca_step(self):
        self.sim.doStep()
        self.update_pref_vel()
        # self.gym_obs = self._get_obs()

        # pref_vel = np.array(self.comp_pref_vel(self.world["agents_id"][0]))

        # need to update for multiple robots
        # need to move this into render() or step()
        # self.world["laserScans"][0] = self.rotate_laser_scan(self.gym_obs['agent_0'], pref_vel)
        # if self.done_test() == True:
            # print('episode_steps,', self.step_count)
        if FLAG_DRAW:
            self.draw_update()

        self.step_count += 1

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
        self.agents_done = [0] * self.numAgents

        self._init_world()

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

        if FLAG_DRAW:
            self.win.destroy()
            self._init_visworld()

        return self._get_obs()

    '''
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
    '''

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
        # time.sleep(self.timeStep)
        desired_time = self.step_count*self.timeStep
        sleep_time = desired_time/self.play_speed - (time.time() - self.t_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    # congested
    # incoming
    # crowd
    # deadlock
    # circle
    # blocks
    CA = Collision_Avoidance_Env(20, "incoming")
    print(CA.run_sim(0))