import rvo2
import random as rnd
from tkinter import *
import time
from math import sin, cos, sqrt, exp
import numpy as np


def init_vis():
    for i in range(numAgents):
        if i == 0:
            visWorld["bot_circles"].append(canvas.create_oval(0, 0, radius, radius, fill="red"))
        else:
            visWorld["bot_circles"].append(canvas.create_oval(0, 0, radius, radius, fill="white"))
        visWorld["vel_lines"].append(canvas.create_line(0, 0, 10, 10, fill="red"))
        visWorld["pref_vel_lines"].append(canvas.create_line(0, 0, 10, 10, fill="green"))



def draw_world():
    for i in range(numAgents):
        scale = pixelsize / envsize
        canvas.coords(visWorld["bot_circles"][i],
                      scale * (sim.getAgentPosition(world["agents"][i])[0] - radius),
                      scale * (sim.getAgentPosition(world["agents"][i])[1] - radius),
                      scale * (sim.getAgentPosition(world["agents"][i])[0] + radius),
                      scale * (sim.getAgentPosition(world["agents"][i])[1] + radius))
        canvas.coords(visWorld["vel_lines"][i],
                      scale * sim.getAgentPosition(world["agents"][i])[0],
                      scale * sim.getAgentPosition(world["agents"][i])[1],
                      scale * (sim.getAgentPosition(world["agents"][i])[0] + 1. * radius *
                               sim.getAgentVelocity(world["agents"][i])[0]),
                      scale * (sim.getAgentPosition(world["agents"][i])[1] + 1. * radius *
                               sim.getAgentVelocity(world["agents"][i])[1]))
        canvas.coords(visWorld["pref_vel_lines"][i],
                      scale * sim.getAgentPosition(world["agents"][i])[0],
                      scale * sim.getAgentPosition(world["agents"][i])[1],
                      scale * (sim.getAgentPosition(world["agents"][i])[0] + 1. * radius *
                               sim.getAgentPrefVelocity(world["agents"][i])[0]),
                      scale * (sim.getAgentPosition(world["agents"][i])[1] + 1. * radius *
                               sim.getAgentPrefVelocity(world["agents"][i])[1]))


def rand_init_agents():
    for i in range(numAgents):
        world["agents"].append(
            sim.addAgent((0, 0), neighborDist, maxNeighbors, timeHorizon, radius, maxSpeed, 1, (0, 0)))

        pos = (rnd.uniform(0, envsize), rnd.uniform(0, envsize))
        angle = rnd.uniform(0, 2 * 3.141592)
        vel = (5 * cos(angle), 5 * sin(angle))

        sim.setAgentPosition(world["agents"][i], pos)
        sim.setAgentPrefVelocity(world["agents"][i], vel)


timeStep = 1/60.
neighborDist = 1.5
maxNeighbors = 5
timeHorizon = 1.5
radius = 0.5  # 2
maxSpeed = 0.4
velocity = 1#(1, 1)

numAgents = 10

world = {}
world["agents"] = []
world["obstacles"] = []

visWorld = {}
visWorld["bot_circles"] = []
visWorld["vel_lines"] = []
visWorld["pref_vel_lines"] = []

sim = rvo2.PyRVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, radius, maxSpeed, velocity)

envsize = 10
rand_init_agents()


print('Simulation has %i agents in it.' %
      (sim.getNumAgents()))

# visualization stuff

# Environmental Specifciation

pixelsize = 700
# envsize = 10
framedelay = 30

win = Tk()
canvas = Canvas(win, width=pixelsize, height=pixelsize, background="#444")
canvas.pack()

init_vis()
draw_world()
win.update_idletasks()
win.update()
time.sleep(timeStep)

# ending visualization stuff

print('Running simulation')

for step in range(2000):
    sim.setAgentPrefVelocity(world["agents"][0], (-1, -1))
    sim.doStep()

    # positions = ['(%5.3f, %5.3f, %5.3f)' % sim.getAgentPosition(agent_no) for agent_no in world["agents"]]
    # print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))

    draw_world()
    win.update_idletasks()
    win.update()
    time.sleep(timeStep)
