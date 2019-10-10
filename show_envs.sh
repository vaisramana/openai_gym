
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
#from IPython.display import display, HTML



def draw_mountain():
    min_x = -1.2
    max_x = 0.6
    x = np.linspace(min_x, max_x, 100)
    y = np.sin(3*x)*.45+.55
    plt.plot(x, y)
    
    my_x_ticks = np.arange(min_x, max_x, 0.1)
    plt.xticks(my_x_ticks)
    
    plt.show()






space_names = ['obs space', 'act space', 'reward range', 'mas steps']
df = pd.DataFrame(columns=space_names)


skip_env_id = ['Defender-v0', 'Defender-v4', 'DefenderDeterministic-v0']

env_specs = gym.envs.registry.all()
for env_spec in env_specs:
    env_id = env_spec.id

    if env_id in skip_env_id:
        print("skip  make %s" %(env_id))
        #continue
        break

    try:
        print("try make %s" %(env_id))
        env = gym.make(env_id)
        print("%s make done" %(env_id))
        observation_space = env.observation_space
        action_space = env.action_space
        reward_range = env.reward_range
        max_episode_steps = None
        if isinstance(env, gym.wrappers.time_limit.TimeLimit):
            max_episode_steps = env._max_episode_steps
        df.loc[env_id] = [observation_space, action_space, reward_range, max_episode_steps]
    except:
        print("%s failed" %(env_id))
        pass



with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

