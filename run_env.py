import argparse
import sys

import gym
from gym import wrappers, logger
import agent


def make_env(env_name):
    env = gym.make(env_name)
    print('observation space = {}'.format(env.observation_space))
    print('action space = {}'.format(env.action_space))
    print('reward range = {}'.format(env.reward_range))
    if isinstance(env, gym.wrappers.time_limit.TimeLimit):
         print('max episode steps = {}'.format(env._max_episode_steps))
    return env






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = make_env(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = './random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = agent.random(env.action_space)
    #agent = agent.bespoke(env.action_space)

    episode_count = 10
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        step = 0
        episode_reward = 0
        while True:
            action = agent.act(ob, reward, done)
            next_ob, reward, done, _ = env.step(action)
            print("%d/%d ob %r rew %r done %r -> act %r -> ob %r rew %r done %r" %(step, i, ob, reward, done, action, next_ob, reward, done))
            step += 1
            episode_reward += reward
            agent.learn(ob, action, reward, done)
            ob = next_ob
            if done:
                print("episode %d done with episode reward %d" %(i, episode_reward));
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
