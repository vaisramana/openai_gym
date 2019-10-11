import argparse
import sys

import gym
from gym import wrappers, logger
import agent.common
import agent.mountaincar


def make_env(env_name):
    env = gym.make(env_name)
    print('observation space = {}'.format(env.observation_space))
    print('action space = {}'.format(env.action_space))
    print('reward range = {}'.format(env.reward_range))
    if isinstance(env, gym.wrappers.time_limit.TimeLimit):
         print('max episode steps = {}'.format(env._max_episode_steps))
    return env




def episode(env, agent, render=False, train=False):
    episode_reward = 0. 
    reward = 0
    observation = env.reset() 
    step = 0
    while True: 
        if render: 
            env.render() 
        action = agent.act(observation)
        next_observation, reward, done, _ = env.step(action) 
        #print("%d ob %r rew %r done %r -> act %r -> ob %r rew %r done %r" %(step, observation, reward, done, action, next_observation, reward, done))
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done) 
        if done:
            print("episode done with episode reward %d" %(episode_reward));
            break
        observation = next_observation
        step += 1
    return episode_reward





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = make_env(args.env_id)

    env.seed(0)
    #agent = agent.common.random(env.action_space)
    agent = agent.mountaincar.bespoke(env.action_space)

    episode_count = 3
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        step = 0
        #episode_reward = episode(env, agent, render=False, train=True)
        episode_reward = episode(env, agent, render=True, train=True)

    # Close the env and write monitor result info to disk
    env.close()
