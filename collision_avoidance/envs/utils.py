from math import sqrt
from numpy import argsort
import numpy as np

def line_intersection(L1,L2) :
    p0,p1 = L1[0],L1[1]
    p2,p3 = L2[0],L2[1]

    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]

    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0 : 
        return float('inf'),(0,0) # collinear

    denom_is_positive = denom > 0
    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]
    s_numer = s10_x * s02_y - s10_y * s02_x

    if (s_numer < 0) == denom_is_positive :
        return float('inf'),(0,0) # no collision

    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denom_is_positive : 
        return float('inf'),(0,0) # no collision

    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive : 
        return float('inf'),(0,0) # no collision

    # collision detected
    t = t_numer / denom

    pos_x = p0[0] + (t * s10_x)
    pos_y = p0[1] + (t * s10_y)
    d = sqrt(pos_x*pos_x+pos_y*pos_y)

    return d,(pos_x,pos_y)


def comp_laser(laser_lines, lines_with_vel):
    result = []
    for laser_line in laser_lines:
        # print('--------------------------------------------')
        # print('laser_line',laser_line)
        dists, intersections = [],[]

        for line in lines_with_vel:
            d, i = line_intersection(laser_line,line[0])
            dists.append(d)
            intersections.append(i)

        # print('lines_with_vel',lines_with_vel)
        # print('dists',dists)
        min_dist_index = argsort(dists)[0]
        # print('min_dist_index',min_dist_index)
        min_dist = dists[min_dist_index]
        min_dist_inter = intersections[min_dist_index]
        if min_dist_inter == (0,0):
            min_dist_vel = (0,0)
        else:
            min_dist_vel = lines_with_vel[min_dist_index][1]
        # print('min_dist',min_dist)
        # print('min_dist_inter',min_dist_inter) # (0.21588836319353769, 0.5212006143803669)
        # print('min_dist_vel',min_dist_vel) # (0.9885764122009277, 0.15071740746498108)

        result.append((min_dist_inter,min_dist_vel))

    return result


def comp_laser_with_rot(laser_lines, lines_with_vel, orientation):
    # print("TESTING HERE:", lines_with_vel)
    # print(laser_lines)
    # input()

    # negative so that we convert back
    theta = -np.arctan2(orientation[1], orientation[0])

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

    new_lines_with_vel = []
    for loc_vel in lines_with_vel:
        loc1 = np.array(loc_vel[0][0])
        loc2 = np.array(loc_vel[0][1])
        loc1_vel = loc1 + np.array(loc_vel[1])

        loc1 = rot@loc1
        loc2 = rot@loc2
        loc1_vel = rot@loc1_vel
        vel = loc1_vel - loc1

        new_lines_with_vel.append([[loc1, loc2], vel])
    
    '''
    #just multiply for once, maybe faster?
    #turn out to be slower...
    all_points = []
    for loc_vel in lines_with_vel:
        loc1 = loc_vel[0][0]
        loc2 = loc_vel[0][1]
        loc1_vel = (loc1[0] + loc_vel[1][0], loc1[1] + loc_vel[1][1])

        all_points.append(loc1) 
        all_points.append(loc2) 
        all_points.append(loc1_vel) 
    all_points = np.array(all_points)
    roted_all_points = rot@(all_points.T)
    roted_all_points = all_points.reshape(-1,6)

    for ps in roted_all_points:
        new_lines_with_vel.append([[(ps[0],ps[1]),(ps[2],ps[3])],(ps[4]-ps[0],ps[5]-ps[1])])
    '''

    result = []
    for laser_line in laser_lines:
        # print('--------------------------------------------')
        # print('laser_line',laser_line)
        dists, intersections = [],[]

        for line in new_lines_with_vel:
            d, i = line_intersection(laser_line,line[0])
            dists.append(d)
            intersections.append(i)

        # print('lines_with_vel',lines_with_vel)
        # print('dists',dists)
        min_dist_index = argsort(dists)[0]
        # print('min_dist_index',min_dist_index)
        min_dist = dists[min_dist_index]
        min_dist_inter = intersections[min_dist_index]
        if min_dist_inter == (0,0):
            min_dist_vel = (0,0)
        else:
            min_dist_vel = new_lines_with_vel[min_dist_index][1]
        # print('min_dist',min_dist)
        # print('min_dist_inter',min_dist_inter) # (0.21588836319353769, 0.5212006143803669)
        # print('min_dist_vel',min_dist_vel) # (0.9885764122009277, 0.15071740746498108)

        result.append((min_dist_inter,min_dist_vel))

    return result