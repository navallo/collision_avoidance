from ALAN_true import Collision_Avoidance_Sim

class MCMC_train():
    def __init__(self, numAgents=50, scenario="crowd", numRounds=10):
        self.numAgents = numAgents
        self.scenario = scenario
        self.numRounds = numRounds

        simulator = Collision_Avoidance_Sim(self.numAgents, self.scenario, True)
        simulator.run_sim(1)

        # init action set

        # init evaluation
        # init temp
        for i in range(numAgents):
            print()
            # select modification
            # apply modification
            # evaluate
            # update aciton set
            # update temp
        # return action set




if __name__ == "__main__":
    # congested
    # crowd
    # deadlock
    # circle
    # blocks
    # CA = Collision_Avoidance_Sim(20, "blocks", True)
    # print(CA.run_sim(1))
    MCMC_train(numAgents=50, scenario="crowd", numRounds=10)

