from math import sin, cos, sqrt, exp, pi
import time
from random import uniform

from tkinter import *
import rvo2

from collision_avoidance.envs.utils import *

class Collision_Avoidance_Sim():

    def __init__(self, numAgents=50, scenario="crowd", online_actions=None, visualize=True):

        # ORCA config
        self.timeStep = 1/60.
        self.neighborDist = 5
        self.maxNeighbors = 10
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
        self.default_online_actions = [(1, 0),
            (0.70711, 0.70711),
            (0, 1),
            (-0.70711, 0.70711),
            (-1, 0),
            (-0.70711, -0.70711),
            (0, -1),
            (0.70711, -0.70711)]
        self.online_actions = self.default_online_actions

        # Update online actions if necessary
        if online_actions is None:
            self.online_actions = self.default_online_actions
        else:
            self.online_actions = online_actions

        self.gamma = 0.6
        self.timewindow = 2
        self.online_temp = 0.2

        # world config
        self.numAgents = numAgents
        self.scenario = scenario

        self.pixelsize = 1000
        self.envsize = self.radius*numAgents

        self.step_count = 0
        self.max_step = int((10/self.timeStep) * self.numAgents)

        self.agents_done = [0] * self.numAgents
        self.agents_time = [self.max_step*self.timeStep] * self.numAgents

        self._init_world()
        self.visualize = visualize
        if visualize:
            self._init_visworld()
            self.visualize_action_space(self.online_actions)

        # Data collection
        self.min_TTime = 0
        self.TTime = 0

        # Start timer (used for simulation video)
        self.t_start = time.time()
        self.play_speed = 4

    # Reset for another simulation run
    def reset(self, online_actions=None):
        # ORCA config
        self.sim = rvo2.PyRVOSimulator(timeStep=self.timeStep,
                                       neighborDist=self.neighborDist,
                                       maxNeighbors=self.maxNeighbors,
                                       timeHorizon=self.timeHorizon,
                                       timeHorizonObst=self.timeHorizon,
                                       radius=self.radius,
                                       maxSpeed=self.maxSpeed)

        # Update online actions if necessary
        if online_actions is None:
            self.online_actions = self.default_online_actions
        else:
            self.online_actions = online_actions

        # World config
        self.step_count = 0
        self.agents_done = [0] * self.numAgents
        self.agents_time = [self.max_step * self.timeStep] * self.numAgents

        self._init_world()
        if self.visualize:
            self.win.destroy()
            self._init_visworld()

    # Run the simulation
    def run_sim(self, mode=1):
        # Check if valid mode is given
        if mode != 1 & mode != 0:
            mode = 1

        # Step through simulation
        self.t_start = time.time()
        success = False
        for i in range(self.max_step):
            if mode == 1:
                self.online_step()
            else:
                self.orca_step()
            # Increment steps
            self.step_count += 1
            success = self.done_test()
            if success:
                break

        # Calculate TTime
        total_time = self.step_count*self.timeStep
        times = np.array(self.agents_time)
        ave_time = np.average(times)
        std_time = np.std(times, 0)
        self.TTime = ave_time + 3*std_time
        return success, total_time, self.TTime, self.min_TTime

    # Initialize the world
    def _init_world(self):

        # Setup world dictionary
        self.world = {}
        self.world["agents_id"] = []
        self.world["obstacles_vertex_ids"] = []
        self.world["laserScans"] = []
        self.world["action_weights"] = []
        self.world["action_times"] = []
        self.world["targets_pos"] = []

        # Initialize specific scenario
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

        # Calculate min TTime
        min_times = []
        for i in range(len(self.world["agents_id"])):
            agent = self.world["agents_id"][i]
            start = self.sim.getAgentPosition(agent)
            goal = self.world["targets_pos"][i][0]
            dist = sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)
            min_times.append(self.maxSpeed*dist)
        times = np.array(min_times)
        ave_time = np.average(times)
        std_time = np.std(times, 0)
        self.min_TTime = ave_time + 3*std_time

    # Initialize congested world
    def _init_world_congested(self):
        # Adjust environment size
        self.envsize = sqrt(2 * self.radius * self.numAgents) * 3
        # Add agents
        for i in range(self.numAgents):
            pos = (uniform(self.envsize * 0.2, self.envsize), uniform(0, self.envsize))
            angle = uniform(0, 2 * pi)
            pref_vel = (cos(angle), sin(angle))
            self._init_add_agents(pos, pref_vel)

            # Set target
            target_pos = ((0.1 * self.envsize - 1.0, self.envsize / 2),
                          (0.1 * self.envsize - self.envsize, self.envsize / 2))
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

        self.update_pref_vel()

        # Border obstacle
        self._init_add_obstacles((-self.envsize, 0.0), (-self.envsize, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        # Internal obstacles
        self._init_add_obstacles((0.1 * self.envsize, 0.0),
                                 (0.1 * self.envsize + 0.5, 0.0),
                                 (0.1 * self.envsize + 0.5, self.envsize / 2 - 1.25 * self.radius),
                                 (0.1 * self.envsize, self.envsize / 2 - 1.25 * self.radius))

        self._init_add_obstacles((0.1 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (0.1 * self.envsize + 0.5, self.envsize / 2 + 1.25 * self.radius),
                                 (0.1 * self.envsize + 0.5, self.envsize),
                                 (0.1 * self.envsize, self.envsize))

        self.sim.processObstacles()

    # Initialize incoming world
    def _init_world_incoming(self):
        # Adjust environment size
        self.envsize = sqrt(2 * self.radius * self.numAgents) * 10

        # Add single agent
        pos = (0.1 * self.envsize, self.envsize / 2)
        angle = uniform(0, 2 * pi)
        pref_vel = (cos(angle), sin(angle))
        self._init_add_agents(pos, pref_vel)

        # Set target
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

            # Increment position
            y_pos += y_inc
            if(y_pos > y_start + y_inc*agent_block_len):
                x_pos += x_inc
                y_pos = y_start


        self.update_pref_vel()

        # Border obstacle
        self._init_add_obstacles((0.0, 0.0), (0.0, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        self.sim.processObstacles()

    # Initialize crowd world
    def _init_world_crowd(self):
        # Adjust environment size
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

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

        self.update_pref_vel()

        # Border obstacle
        self._init_add_obstacles((0.0, 0.0), (0.0, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        self.sim.processObstacles()

    # Initialize circle world
    def _init_world_circle(self):
        # Adjust environment size
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

            # Set target
            target = (self.envsize/2 + circle_radius*cos(theta + pi),
                      self.envsize/2 + circle_radius*sin(theta + pi))
            target_pos = (target, target)
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

            theta += theta_inc

        self.update_pref_vel()

        # Border obstacle
        self._init_add_obstacles((0.0, 0.0), (0.0, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        self.sim.processObstacles()

    # Initialize blocks world
    def _init_world_blocks(self):
        # Adjust environment size
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

            # Set target
            target = (self.envsize - 1.5 * self.radius, y_pos)
            target_pos = (target, target)
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))

            y_pos += y_inc

        self.update_pref_vel()

        # Border obstacle
        self._init_add_obstacles((0.0, 0.0), (0.0, self.envsize),
                                 (self.envsize, self.envsize), (self.envsize, 0.0))

        # Blocks
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

    # Initialize deadlock world
    def _init_world_deadlock(self):
        # Adjust environment size
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

            # Set target
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

            # Set target
            target_pos = ((0.1 * self.envsize, self.envsize / 2),
                          (0.1 * self.envsize - self.envsize, self.envsize / 2))
            self.world["targets_pos"].append(target_pos)

            self.world["action_weights"].append([0.0] * len(self.online_actions))
            self.world["action_times"].append([0.0] * len(self.online_actions))
            pos_x += x_inc

        self.update_pref_vel()

        # Border obstacle
        self._init_add_obstacles((-self.envsize, 0.0), (-self.envsize, self.envsize),
                                 (2 * self.envsize, self.envsize), (2 * self.envsize, 0.0))

        # Internal obstacles

        # Left funnel
        self._init_add_obstacles((0.0, 0.0),
                                 (0.0 + 0.5, 0.0),
                                 (0.2 * self.envsize + 0.5, self.envsize / 2 - 1.25 * self.radius),
                                 (0.2 * self.envsize, self.envsize / 2 - 1.25 * self.radius))

        self._init_add_obstacles((0.0, self.envsize),
                                 (0.2 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (0.2 * self.envsize + 0.5, self.envsize / 2 + 1.25 * self.radius),
                                 (0.0 + 0.5, self.envsize))

        # Right funnel
        self._init_add_obstacles((self.envsize - 0.5, 0.0),
                                 (self.envsize, 0.0),
                                 (0.8 * self.envsize, self.envsize / 2 - 1.25 * self.radius),
                                 (0.8 * self.envsize - 0.5, self.envsize / 2 - 1.25 * self.radius))

        self._init_add_obstacles((self.envsize - 0.5, self.envsize),
                                 (0.8 * self.envsize - 0.5, self.envsize / 2 + 1.25 * self.radius),
                                 (0.8 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (self.envsize, self.envsize))

        # Tube
        self._init_add_obstacles((0.2 * self.envsize, (self.envsize / 2 - 1.25 * self.radius) - 0.5),
                                 (0.8 * self.envsize, (self.envsize / 2 - 1.25 * self.radius) - 0.5),
                                 (0.8 * self.envsize, self.envsize / 2 - 1.25 * self.radius),
                                 (0.2 * self.envsize, self.envsize / 2 - 1.25 * self.radius))

        self._init_add_obstacles((0.2 * self.envsize, (self.envsize / 2 + 1.25 * self.radius) + 0.5),
                                 (0.2 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (0.8 * self.envsize, self.envsize / 2 + 1.25 * self.radius),
                                 (0.8 * self.envsize, (self.envsize / 2 + 1.25 * self.radius) + 0.5))

        self.sim.processObstacles()

    # Initializes and adds an agent
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

        self.sim.setAgentPrefVelocity(agent_id, pref_vel)

    # Initializes and adds an obstacle
    def _init_add_obstacles(self, upper_left, upper_right, bottom_right, bottom_left):
        verticals = []
        vertical_id = self.sim.addObstacle([upper_left, upper_right, bottom_right, bottom_left])
        for j in range(4):
            verticals.append(vertical_id)
            vertical_id = self.sim.getNextObstacleVertexNo(vertical_id)
        self.world["obstacles_vertex_ids"].append(verticals)

    # Updates the preferred velocity to point at the goal
    def update_pref_vel(self):
        for i in range(self.numAgents):
            pref_vel = self.comp_pref_vel(i)
            self.sim.setAgentPrefVelocity(self.world["agents_id"][i], pref_vel)

    # Compute preferred velocity pointing at the goal
    def comp_pref_vel(self, agent_id):
        pos = self.sim.getAgentPosition(self.world["agents_id"][agent_id])
        target_pos = self.world["targets_pos"][agent_id][0]
        angle = np.arctan2(target_pos[1]-pos[1],target_pos[0]-pos[0])
        pref_vel = (cos(angle), sin(angle))

        return pref_vel

    # Initialize the visualization world
    def _init_visworld(self):
        # Setup the window
        self.win = Tk()
        self.win.title(self.scenario + ": " + str(self.numAgents))
        self.canvas = Canvas(self.win, width=self.pixelsize, height=self.pixelsize, background="#eaeaea")
        self.canvas.pack()

        # Setup visWorld dictionary
        self.visWorld = {}
        self.visWorld["bot_circles_id"] = []
        self.visWorld["vel_lines_id"] = []
        self.visWorld["pref_vel_lines_id"] = []

        # Add targets
        self.visWorld["targets_id"] = []
        for i in range(len(self.world["targets_pos"])):
            self.visWorld["targets_id"].append(self.canvas.create_oval(
                0, 0, self.radius, self.radius, outline='', fill="blue"))

        # Add Agents
        for i in range(len(self.world["agents_id"])):
            self.visWorld["bot_circles_id"].append(self.canvas.create_oval(
                -self.radius, -self.radius, self.radius, self.radius, outline='', fill="yellow"))
            self.visWorld["vel_lines_id"].append(self.canvas.create_line(
                    0, 0, self.radius, self.radius, arrow=LAST, width=2, fill="red"))
            self.visWorld["pref_vel_lines_id"].append(self.canvas.create_line(
                    0, 0, self.radius, self.radius, arrow=LAST, width=2, fill="green"))

        # Add Obstacles
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

    # Checks if all the agents are done
    def done_test(self):
        # Iterate through all agents
        for i in range(self.numAgents):
            agent_id = self.world["agents_id"][i]
            if self.agents_done[i] == 0:
                # Check if agent has reached goal
                pos = self.sim.getAgentPosition(self.world["agents_id"][agent_id])
                t_pos = self.world["targets_pos"][agent_id][0]
                if sqrt((pos[0] - t_pos[0])**2 + (pos[1] - t_pos[1])**2) < 2 * self.radius:
                    # Set done flag for agent
                    self.agents_done[agent_id] = 1
                    # Set completion time for agent
                    self.agents_time[agent_id] = self.step_count*self.timeStep
                    # Update target of agent
                    self.world["targets_pos"][agent_id] = (self.world["targets_pos"][agent_id][1],
                                                           self.world["targets_pos"][agent_id][1])
        if 0 not in self.agents_done:
            return True
        else:
            return False

    # Simulation step using Online learning ALAN
    def online_step(self):
        # Setup
        action_vels = []
        pref_vels = []
        action_ids = []

        # Choose an action for each agent
        for i in range(self.numAgents):
            agent_id = self.world["agents_id"][i]

            # Softmax selection
            weights = np.array([i for i in self.world["action_weights"][agent_id]])
            ps = np.exp(weights / self.online_temp)
            ps /= np.sum(ps)

            action_id = int(np.random.choice(len(self.online_actions), 1, p=ps))
            action = self.online_actions[action_id]
            action_ids.append(action_id)

            pref_vel = np.array(self.comp_pref_vel(agent_id))
            pref_vels.append(pref_vel)

            # Convert action to agents reference frame
            goal_theta = np.arctan2(pref_vel[1], pref_vel[0])
            theta = np.arctan2(action[1], action[0])
            goal_theta += theta
            action_vel = (cos(goal_theta), sin(goal_theta))
            action_vels.append(action_vel)

            self.sim.setAgentPrefVelocity(agent_id, (float(action_vel[0]), float(action_vel[1])))

        # Run the ORCA simulation
        self.sim.doStep()
        if self.visualize:
            self.draw_update()

        # Evaluate the actions of each agent
        for i in range(self.numAgents):
            pref_vel = pref_vels[i]
            action_vel = action_vels[i]
            action_id = action_ids[i]

            agent_id = self.world["agents_id"][i]

            orca_vel = self.sim.getAgentVelocity(agent_id)

            # Calculate reward
            R_goal = np.dot(orca_vel, pref_vel)
            R_polite = np.dot(orca_vel, action_vel)
            R = self.gamma*R_goal + (1-self.gamma)*R_polite

            # Increment time window of action weights
            for act_id in range(len(self.online_actions)):
                self.world["action_times"][agent_id][act_id] += self.timeStep
                if self.world["action_times"][agent_id][act_id] >= self.timewindow:
                    self.world["action_times"][agent_id][act_id] = 0
                    self.world["action_weights"][agent_id][act_id] = 0

            # Update weight
            self.world["action_weights"][agent_id][action_id] = R

    # Simulation step using only ORCA
    def orca_step(self):
        self.sim.doStep()
        self.update_pref_vel()

        if self.visualize:
            self.draw_update()

    # Draws the state of the world to the window
    def draw_world(self):
        scale = self.pixelsize / self.envsize

        # Draw targets
        for i in range(len(self.world["targets_pos"])):
            self.canvas.coords(self.visWorld["targets_id"][i],
                               scale * (self.world["targets_pos"][i][0][0] - self.radius),
                               scale * (self.world["targets_pos"][i][0][1] - self.radius),
                               scale * (self.world["targets_pos"][i][0][0] + self.radius),
                               scale * (self.world["targets_pos"][i][0][1] + self.radius))

        # Draw agents
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

        # Draw obstacles
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

    # Updates the visualization
    def draw_update(self):
        # Draw world and update window
        self.draw_world()
        self.win.update_idletasks()
        self.win.update()

        # Calculate display delay
        desired_time = self.step_count*self.timeStep
        sleep_time = desired_time/self.play_speed - (time.time() - self.t_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def visualize_action_space(self, actions):
        win = Tk()
        win.title("Action space")
        pixelsize = 1000
        canvas = Canvas(win, width=pixelsize, height=pixelsize, background="white")
        canvas.pack()

        half_window = pixelsize/2
        line_width = int(0.01*pixelsize)
        for action in actions:
            canvas.create_line(half_window,
                               half_window,
                               half_window + 0.9*half_window*action[0],
                               half_window + 0.9*half_window*action[1],
                               arrow=LAST, width=line_width, fill="blue")

        radius = 0.3 * half_window
        canvas.create_oval(half_window - radius,
                           half_window - radius,
                           half_window + radius,
                           half_window + radius,
                           outline="black", fill="yellow")

        win.update_idletasks()
        win.update()

if __name__ == "__main__":
    # scenarios:
    # congested
    # crowd
    # deadlock
    # circle
    # blocks
    # incoming

    # comment or uncomment these sections to run different simulations

    # CA = Collision_Avoidance_Sim(numAgents=50, scenario="congested",
    #                              online_actions=[(1, 0), (0.9009108952317388, 0.4340041000414015), (0.8425621872144076, 0.5385990722944792), (-0.9747090437965987, -0.2234776945046645), (-0.7297166474031178, 0.6837496723968165), (0.3099284486326036, -0.9507598838445949), (0.9990288702365868, 0.04406037260179406), (-0.2961053847313458, -0.9551552759280041), (-0.9898935365354221, 0.14181250412215018)],
    #                              visualize=True)

    # CA = Collision_Avoidance_Sim(numAgents=50, scenario="deadlock",
    #                              online_actions=[(1, 0), (-0.8507885983503763, -0.5255080978605392)],
    #                              visualize=True)

    # CA = Collision_Avoidance_Sim(numAgents=21, scenario="incoming",
    #                              online_actions=[(1, 0), (-0.8773828831791611, -0.4797908672580403), (0.1520704393368215, -0.988369658316111), (0.15098864856953262, 0.9885354965822655), (0.49267960206794637, -0.8702107846413821)],
    #                              visualize=True)

    CA = Collision_Avoidance_Sim(numAgents=100, scenario="circle",
                                 online_actions=[(1, 0), (-0.946001067452245, -0.3241635087100537), (0.9651519613079926, -0.2616900677964968)],
                                 visualize=True)

    # CA = Collision_Avoidance_Sim(numAgents=40, scenario="crowd",
    #                              #online_actions=[(1, 0), (0.06130798855686512, -0.9981188959934139), (0.6957331792402002, -0.718300315539624), (0.2199481143650689, 0.9755115719391804), (-0.34342388205007957, -0.9391805136594631), (-0.5032837686361407, -0.8641211999641044), (0.23892114215448593, 0.9710389733844857), (-0.9579667593692613, -0.28687922187491344), (-0.6767380373137093, 0.7362238985884584)],
    #                              visualize=True)

    # run sim with 0 to use basic ORCA, run sim with 1 to get ALAN
    print(CA.run_sim(1))


    # Collecting data (used for report)
    # total = 0
    # num = 3
    # CA = Collision_Avoidance_Sim(numAgents=20, scenario="blocks",
    #                              online_actions=[(1, 0), (0.47793668363831737, -0.8783942887068465), (-0.9934856438219436, 0.11395734078899142), (-0.8813346776537921, -0.47249252477143616), (-0.9703087227605884, -0.2418697635809723), (0.8163336987112819, 0.577580550528113), (0.1553219334597776, -0.987863906105652), (-0.8927531532165186, -0.450546121303872)],
    #                              visualize=True)
    # for i in range(num):
    #     CA.reset()
    #     finished, total_time, TTime, min_TTime = CA.run_sim(1)
    #     total += TTime
    #
    # print(total/num)