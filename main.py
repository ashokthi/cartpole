import gym
import tensorflow as tf
import numpy as np
from rlagents import MonteCarlo, Sarsa, Sarsa20


agent = Sarsa20() #options are MonteCarlo, Sarsa, and Sarsa20
env = gym.make('CartPole-v0')
sess = tf.Session()
agent.build(sess)
disc = 0.95

for i_episode in range(500):
    obs = env.reset()
    obsAll = []
    obsAll.append(obs)
    rewardsAll = []
    actionsAll = []
    returnsAll = []

    for t in range(200):
        env.render()
        action = agent.step(obs,sess)
        obs, reward, done, _ = env.step(action)
        obsAll.append(obs)
        rewardsAll.append(reward)
        actionsAll.append(action)

        if done:
            returnsAll = agent.calcReturns(rewardsAll,obsAll,actionsAll,disc,sess)
            actionsAll = np.expand_dims(actionsAll,1)
            agent.train([obsAll,actionsAll,returnsAll],sess)
            print("Episode %d, Length %d" % (i_episode,t))

            break