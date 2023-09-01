# play with rl gyms
import gymnasium as gym

n_envs = 4

if __name__ == '__main__':
    # envs = gym.vector.SyncVectorEnv([gym.make('LunarLander-v2') for i in range(n_envs)])
    envs = gym.vector.SyncVectorEnv([lambda: gym.make('LunarLander-v2') for i in range(n_envs)])

    def ll_policy(observation):
        actions = envs.action_space.sample()
        return actions

    num_steps = 256
    trajectory = []
    splits = []
    rewards = []
    gamma = 0.99
    # last_obs = None

    # rollout
    last_obs = envs.reset()
    for step in range(num_steps):
        action = ll_policy(last_obs)
        next_obs, reward, dones, _, info = envs.step(action)
        trajectory.append((last_obs, action, next_obs))
        rewards.append(reward)
        splits.append(dones)
        last_obs = next_obs
        # if terminated:
        #    last_obs = envs.reset()

    print(rewards)
    print(splits)

    gains = []
    gain = 0
    # calculate accumulated rewards & advantage
    for step in reversed(range(num_steps)):
        # if episode terminates then set value to 0,
        # else set it via bootstrap (value function)
        gain = rewards[step] + gamma * gain
        gains.insert(0, gain)

    # print(gains)