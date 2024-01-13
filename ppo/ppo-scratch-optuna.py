import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
import optuna

DEBUG = False

# use MPS on mac os
device_name = 'cpu'
# device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device_name)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# define an (actor-critic) agent with policy / value network 
class MlpAgent(nn.Module):
    def __init__(self, in_size, h1_size, h2_size, out_size):
        super().__init__()
        self.actor = nn.Sequential(layer_init(nn.Linear(in_size, h1_size)),
                                nn.Tanh(),
                                layer_init(nn.Linear(h1_size, h2_size)),
                                nn.Tanh(),
                                layer_init(nn.Linear(h2_size, out_size)))
        self.critic = nn.Sequential(layer_init(nn.Linear(in_size, h1_size)),
                                nn.Tanh(),
                                layer_init(nn.Linear(h1_size, h2_size)),
                                nn.Tanh(),
                                layer_init(nn.Linear(h2_size, 1)))
    
    def get_action_value(self, x, a=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if a == None:
            a = probs.sample()
        v = self.critic(x)
        return a, probs.log_prob(a), probs.entropy(), v
    
    def get_values(self, x):
        return self.critic(x)

def define_model(trial, in_size, out_size):
    mlp_h1_size = 128
    mlp_h2_size = 64
    agent = MlpAgent(in_size, mlp_h1_size, mlp_h2_size, out_size).to(device)

    return agent

MAX_STEPS = 5e3
INIT_GAIN = -1000
ENV_ID = 'LunarLander-v2'

def objective(trial: optuna.Trial):
    env_id = ENV_ID
    evaluation = INIT_GAIN # TODO check (for optuna)

    # mini_batch_size = 4
    max_steps = MAX_STEPS
    num_steps = trial.suggest_int("num_steps", 64, 1024) # 512 # 128, 256
    n_envs = trial.suggest_int("n_epochs", 2, 8) # 4
    gamma = 0.99
    n_epochs = trial.suggest_int("n_epochs", 2, 16) # 8
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2) # 1e-3
    clip_coef = trial.suggest_float("clip_coef", 0.01, 1) # 0.2
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 1) # 0.03
    vf_coef = trial.suggest_float("clip_coef", 0.01, 1) # 0.5
    anneal_lr = True

    # create the environment (vector environment)
    # envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)])
    envs = gym.vector.make(env_id, n_envs)
    in_size = np.array(envs.single_observation_space.shape).prod()
    out_size = envs.single_action_space.n
    agent = define_model(trial, in_size, out_size)
    optimizer = optim.Adam(agent.parameters(), lr = learning_rate)

    # allocate replay buffer ( for a batch )
    #   to store history of: observations, actions taken, action prob, rewards, GAE, next state
    obs = torch.zeros((num_steps, n_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, n_envs) + envs.single_action_space.shape).to(device)
    action_probs = torch.zeros((num_steps, n_envs)).to(device)
    rewards = torch.zeros((num_steps, n_envs)).to(device)
    # next_obs = []
    terminates = torch.zeros((num_steps, n_envs)).to(device)
    gains = torch.zeros((num_steps, n_envs)).to(device)
    advantages = torch.zeros((num_steps, n_envs)).to(device)
    values = torch.zeros((num_steps, n_envs)).to(device)

    # training loop
    num_batches = int(max_steps // (num_steps * n_envs))
    for batch in range(num_batches):
        frac = 1
        if anneal_lr:
            frac = 1.0 - (batch - 1.0) / num_batches
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        # print(f'start training with batch {batch} / {num_batches} w/ learning rate {lrnow} ... ')
        # rollout num_steps and store the trajectory into replay buffer
        ob, _ = envs.reset()
        ob = torch.Tensor(ob).to(device)
        for step in range(num_steps):
            with torch.no_grad():
                a, r_log_probs, r_entropy, r_value = agent.get_action_value(ob)
                values[step] = r_value.flatten()
            next_ob, r, done, _, _ = envs.step(a.cpu().numpy())
            obs[step] = ob
            actions[step] = a
            action_probs[step] = r_log_probs
            rewards[step] = torch.Tensor(r).to(device)
            terminates[step] = torch.Tensor(done).to(device)
            # next_obs.append(next_ob)
            ob = torch.Tensor(next_ob).to(device)

        #   calculate the gains and advantages (not GAE)
        #   if not the end of episode, then bootstrap with value function, otherwise 0
        #   TODO need to calculate V(st) on V(st+1) (shift left 1 place). or will create huge bias.
        with torch.no_grad():
            for t in reversed(range(num_steps)):
                if t < num_steps - 1:
                    gains[t] = rewards[t] + (1 - terminates[t]) * gamma * gains[t + 1]
                else: 
                    gains[t] = rewards[t] + (1 - terminates[t]) * values[t]
            advantages = gains - values
            # TODO also implement for GAEs
        
            # calculate the mean and std of gains
            shifted_terminates = torch.zeros_like(terminates)
            shifted_terminates[1:] = terminates[:num_steps - 1]
            shifted_terminates[0] = 1
            raw_episodes_gains = (shifted_terminates * gains).flatten()
            episodes_gains = []
            for g in raw_episodes_gains:
                if g != 0:
                    episodes_gains.append(g.item())
            t_episode_gains = torch.tensor(episodes_gains)
            # print(f'The average gain of current batch is {t_episode_gains.mean()} +/- {t_episode_gains.std()}')

            succeeded = ((torch.Tensor(rewards) + 100) // 200).sum().item()
            print(f'{terminates.sum()} episodes terminated with {succeeded} successful landings')
            # DEBUG only: output rewards and gains, check if gains are computed correctly
            if DEBUG and succeeded > 0:
                with open('rollout-rewards.txt', 'w') as f:
                    f.write('Rewards: \n')
                    for t in range(num_steps):
                        for r in rewards[t]:
                            f.write('{:.2f}'.format(r) + ', ')
                        f.write('    |    ')
                        for g in gains[t]:
                            f.write('{:.2f}'.format(g) + ', ')
                        f.write('    |    ')
                        for tm in terminates[t]:
                            f.write(f'{tm.item()}' + ', ')
                        f.write('\n')
                break
            # print(obs, rewards, gains)

        #   randomly pick a mini batch from replay buffer
        mb_indices = [i for i in range(num_steps)]
        np.random.shuffle(mb_indices)
        for _ in range(n_epochs):
            for i in mb_indices:
                # TODO add mini batch support later; single example training here
                # batch = mb_indices[b * mini_batch_size : (b + 1) * mini_batch_size]

                #   calculate the probability of action under new policy
                #    - by passing the observation through the actor-critic network
                a = actions[i]
                _, a_logprob, a_entropy, v = agent.get_action_value(obs[i], a)

                #   calculate the portion of probabilities of action under new and old policy
                p_theta_log = a_logprob - action_probs[i]
                p_theta = torch.exp(p_theta_log)

                #   calculate the clipped policy surrogate objective function
                adv = advantages[i]
                loss_cp1 = - adv * p_theta
                loss_cp2 = - adv * torch.clamp(p_theta, 1 - clip_coef, 1 + clip_coef)

                #   calculate the policy gradient loss ( multiple clipped surrogate fuction with GAE )
                pg_loss = torch.max(loss_cp1, loss_cp2).mean()

                #   calculate the value loss
                v_loss = ((v - gains[i]) ** 2).mean() / 2

                #   entropy loss
                entropy_loss = a_entropy.mean()

                #   calculate the actor-critic objective function
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                # step and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # evaluate the model
    eval_max_steps = 500
    eval_rounds = 10
    eval_env = gym.make(env_id, render_mode='rgb_array')
    gains = []
    frames_list = []
    episodes = []
    rewards = []
    terminated = False
    for round in range(eval_rounds):
        # print(f'evaluate round {round}')
        gain = 0
        s, _ = eval_env.reset()
        for t in range(eval_max_steps):
            frames_list.append(eval_env.render())
            a, _, _, _ = agent.get_action_value(torch.Tensor(s).to(device))
            s, r, done, _, _ = eval_env.step(a.cpu().numpy())
            gain = r + gamma * gain
            rewards.append('{:.2f}'.format(r))
            if done:
                episodes.append(t)
                terminated = True
                break
        if terminated:
            terminated = False
        else:
            episodes.append(-1)
        gains.append(gain)
    t_gains = torch.Tensor(gains)
    evaluation = t_gains.mean()

    return evaluation

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))