from ALAN_true import Collision_Avoidance_Sim
from random import uniform
from math import sin, cos, sqrt, exp, pi
import numpy as np
from scipy.stats import norm

class MCMC_trainer():
    def __init__(self, numAgents=50, scenario="crowd", numRounds=10):
        self.numAgents = numAgents
        self.scenario = scenario
        self.numRounds = numRounds

        self.simulator = Collision_Avoidance_Sim(numAgents=self.numAgents, scenario=self.scenario, visualize=False)

        # init action set
        self.actions = [(1, 0), self.random_action()]
        self.actions_opt = self.actions
        # init evaluation
        self.eval = self.evaluate_action(self.actions)
        self.eval_opt = self.eval
        # init temp
        self.init_temp = 0.9
        self.final_temp = 0.1
        self.temp = self.init_temp
        self.delta_temp = (self.final_temp - self.init_temp) / (self.numRounds - 1)

    def train(self):
        for i in range(self.numRounds):
            print("run:", i)
            # select modification
            modification = self.select_modification(self.actions, i)
            # apply modification
            action_dist, new_actions = self.apply_modification(self.actions, modification)
            # evaluate
            new_eval = self.evaluate_action(new_actions, i)
            # update optimal action set
            if new_eval < self.eval_opt:
                self.actions_opt = new_actions
                self.eval_opt = new_eval
            # update expolitory action set
            if uniform(0, 1) < self.symmetric_likelihood(action_dist)*exp((self.eval - new_eval)/self.temp):
                self.actions = new_actions
                self.eval = new_eval
            # update temp
            self.temp -= self.delta_temp

        return self.actions_opt

    def random_action(self):
        angle = uniform(-pi, pi)
        return cos(angle), sin(angle)

    def evaluate_action(self, actions, i=0):
        total_score = 0
        num = 3
        for i in range(num):
            self.simulator.reset(online_actions=actions)
            finished, total_time, TTime, min_TTime = self.simulator.run_sim(mode=1)
            if finished:
                print("finished :)")
            else:
                print("did not finish :(")
            print("Total time:", total_time, "TTime:", TTime, "min TTime:", min_TTime)
            total_score += TTime
        return total_score / num

    def select_modification(self, actions, i):
        # if there is only 1 action then you must add
        if len(actions) <= 1:
            return 2
        return np.random.choice(3, p=[0.8, 0.1, 0.1])

    def apply_modification(self, actions, modification):
        if modification == 2:
            return self.mod_add(actions)
        elif modification == 1:
            return self.mod_remove(actions)
        else:
            return self.mod_edit(actions)

    # randomly edit an action
    def mod_edit(self, actions):
        # choose an action to change
        index = np.random.choice(range(1, len(actions)))
        # make a new action
        angle = np.arctan2(actions[index][1], actions[index][0])
        new_angle = np.random.normal(angle, pi)
        new_action = (cos(new_angle), sin(new_angle))
        # calculate distance from current action to new action
        dist = self.dist(actions[index], new_action)
        # update current action to new action
        actions[index] = new_action
        return dist, actions

    # randomly remove an action
    def mod_remove(self, actions):
        # choose an action to remove
        index = np.random.choice(range(1, len(actions)))
        # remove old action
        old_action = actions[index]
        actions.remove(old_action)
        # find distance to closest action
        min_dist = 10
        for act in actions:
            dist = self.dist(act, old_action)
            if dist < min_dist:
                min_dist = dist
        return min_dist, actions

    # add a new random action
    def mod_add(self, actions):
        # choose an action to change
        index = np.random.choice(range(0, len(actions)))
        # make a new action
        angle = np.arctan2(actions[index][1], actions[index][0])
        new_angle = np.random.normal(angle, pi)
        new_action = (cos(new_angle), sin(new_angle))
        # calculate distance from current action to new action
        dist = self.dist(actions[index], new_action)
        # add new action to actions
        actions.append(new_action)
        return dist, actions

    def dist(self, point1, point2):
        return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def symmetric_likelihood(self, dist):
        std = 0.5
        return norm.pdf(dist) / std




if __name__ == "__main__":
    # congested
    # crowd
    # deadlock
    # circle
    # blocks
    # incoming
    mcmc = MCMC_trainer(numAgents=40, scenario="crowd", numRounds=100)
    actions = mcmc.train()
    f = open("crowd_actions.act", "w+")
    f.write(str(actions))
    f.close()



    # mcmc = MCMC_trainer(numAgents=40, scenario="circle", numRounds=100)
    # actions = mcmc.train()
    # f = open("circle_actions.act", "w+")
    # f.write(str(actions))
    # f.close()




    # mcmc = MCMC_trainer(numAgents=21, scenario="incoming", numRounds=100)
    # actions = mcmc.train()
    # f = open("incoming_actions.act", "w+")
    # f.write(str(actions))
    # f.close()



    # mcmc = MCMC_trainer(numAgents=20, scenario="congested", numRounds=100)
    # actions = mcmc.train()
    # f = open("congested_actions.act", "w+")
    # f.write(str(actions))
    # f.close()



    # mcmc = MCMC_trainer(numAgents=20, scenario="deadlock", numRounds=100)
    # actions = mcmc.train()
    # f = open("deadlock_actions.act", "w+")
    # f.write(str(actions))
    # f.close()


