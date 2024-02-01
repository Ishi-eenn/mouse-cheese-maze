import pygame
import sys, os
import random

import numpy as np
import _pickle as pickle
from grid_world import grid_world
from policy_gradient import network

env=grid_world()
observation=env.reset()
xlen=env.xlen
ylen=env.ylen

pg=network(xlen*ylen,200,4)

reward_sum=0
running_reward = None
prev_x = 0
episode_number=0
while True:
    env.render()
    prev_pos=observation
    x = np.zeros(shape=(xlen* ylen), dtype=np.int64)
    x[observation[1]*xlen + observation[0]] = 1
    x[env.reward_pos[1]*xlen + env.reward_pos[0]] = 1


    #今のネズミの位置
    prev_pos = observation


    #チーズの位置
    prev_reward_pos = env.reward_pos

    aprob=pg.forward(x)
    action = pg.select_action(aprob)
    observation,reward,done=env.step(action)
    reward_pos=env.reward_pos
    agent_pos=observation
    reward_sum += reward
    p_pos=abs(reward_pos[0]-prev_pos[0])+abs(reward_pos[1]-prev_pos[1])
    c_pos=abs(reward_pos[0]-agent_pos[0])+abs(reward_pos[1]-agent_pos[1])
    if p_pos>c_pos:
        reward+=1
    else:
        reward-=1

    pg.record_reward(reward)


    if done:
        episode_number += 1

        if reward_sum != 0:
            pg.backward()

        if episode_number % pg.batch_size == 0:
            pg.update()

        running_reward = reward_sum if running_reward is None \
                       else running_reward * 0.99 + reward_sum * 0.01
        print ('ep %d:  episode reward total was %f. running mean: %f'
             % (episode_number, reward_sum, running_reward))


        reward_sum = 0
        observation = env.reset() # reset env

    pygame.display.flip()
