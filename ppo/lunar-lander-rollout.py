import gymnasium as gym
import numpy as np
import math

ENV_ID = 'LunarLander-v2'
ENV_NUM = 4
ACTION_SPACE = 4
MAX_STEPS = 1000

def pickAction():
    action = math.floor(np.random.random() * ACTION_SPACE)
    return action

def rollout(env_id, num_envs):
    rollout_result = []
    envs = gym.vector.make(env_id, num_envs)
    st, _ = envs.reset()
    for _ in range(0, MAX_STEPS):
        actions = []
        for _ in range(0, ENV_NUM):
            actions.append(pickAction())
        (st1, rewards, terminated, _, _) = envs.step(actions)
        observation = (st, actions, rewards, st1, terminated)
        st = st1
        rollout_result.append(observation)
        if not np.all(terminated == 0):
            break
    return rollout_result

if __name__ == '__main__':
    obs = rollout(ENV_ID, ENV_NUM)
    print(len(obs))

    if len(obs) < MAX_STEPS:
        for ob in obs[-3:]:
            print(ob)

