import argparse
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import math
import gymnasium as gym
import cv2
import os
import optuna

DEBUG = False
TUNING_STEPS = 1e4
MAX_STEPS = 1e6
INIT_GAIN = -1000
ENV_ID = 'LunarLander-v2'

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

def define_model(hype_params, in_size, out_size):
    mlp_h1_size = 128
    mlp_h2_size = 64
    agent = MlpAgent(in_size, mlp_h1_size, mlp_h2_size, out_size).to(device)

    return agent

def objective(trial: optuna.Trial):
    hype_params = {}
    hype_params["evaluation"] = INIT_GAIN # TODO check (for optuna)
    hype_params["env_id"] = ENV_ID

    # mini_batch_size = 4
    hype_params["num_steps"] = trial.suggest_int("num_steps", 1000, 2000) # 512 # 128, 256
    hype_params["n_envs"] = trial.suggest_int("n_envs", 4, 4) # 4
    hype_params["gamma"] = trial.suggest_float("gamma", 0.95, 0.999) # 4
    hype_params["mini_batch"] = trial.suggest_int("mini_batch", 6, 6) # 6
    hype_params["n_epochs"] = trial.suggest_int("n_epochs", 4, 4) # 8
    hype_params["learning_rate"] = trial.suggest_float("learning_rate", 5e-4, 5e-3) # 1e-3
    hype_params["clip_coef"] = trial.suggest_float("clip_coef", 0.1, 0.3) # 0.2
    hype_params["ent_coef"] = trial.suggest_float("ent_coef", 0.2, 0.4) # 0.03
    hype_params["vf_coef"] = trial.suggest_float("vf_coef", 0.4, 0.6) # 0.5
    _, total_gains = ppo(hype_params, True)
    evaluation = total_gains.mean() - total_gains.std()
    return evaluation

def ppo(hype_params: dict, isTuning):
    env_id = hype_params["env_id"]
    learning_rate = hype_params["learning_rate"]
    n_envs = hype_params["n_envs"]
    gamma = hype_params["gamma"]
    mini_batch = hype_params["mini_batch"]
    num_steps = hype_params["num_steps"]
    max_steps = TUNING_STEPS if isTuning else MAX_STEPS
    n_epochs = hype_params["n_epochs"]
    clip_coef = hype_params["clip_coef"]
    ent_coef = hype_params["ent_coef"]
    vf_coef = hype_params["vf_coef"]
    # create the environment (vector environment)
    # envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)])
    anneal_lr = True
    envs = gym.vector.make(env_id, n_envs)
    in_size = np.array(envs.single_observation_space.shape).prod()
    out_size = envs.single_action_space.n
    agent = define_model(hype_params, in_size, out_size)
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
    total_gains = None
    for batch in range(num_batches):
        # for each batch, rollout training data into replay buffer with current policy, and 
        #  train n_epochs times with this replay buffers
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
        #   need to calculate V(st) on V(st+1) (shift left 1 place). or will create huge bias.
        with torch.no_grad():
            for t in reversed(range(num_steps)):
                if t < num_steps - 1:
                    gains[t] = rewards[t] + (1 - terminates[t]) * gamma * gains[t + 1]
                else:
                    # Quality of last episode: Qt = rt + gamma * Vt+1
                    last_value = agent.get_values(ob).flatten()
                    gains[t] = rewards[t] + gamma * (1 - terminates[t]) * last_value # values[t]
            advantages = gains - values
            # TODO also implement for GAEs
        
            # calculate the mean and std of gains
            shifted_terminates = torch.zeros_like(terminates)
            shifted_terminates[1:, :] = terminates[:num_steps - 1,:]
            shifted_terminates[0, :] = 1
            batch_gains = (shifted_terminates * gains).flatten()
            nonzero_batch_gains = batch_gains[batch_gains.nonzero()]
            succeeded = torch.floor((torch.Tensor(rewards) + 100) / 200).sum().item()
            print(f'Batch: {batch}/{num_batches}, Success: {succeeded} / {nonzero_batch_gains.nelement()}, \
                    Return: {nonzero_batch_gains.mean()} +/- {nonzero_batch_gains.std()}')
            if isTuning:
                if total_gains == None:
                    total_gains = nonzero_batch_gains
                else: 
                    torch.cat((total_gains, nonzero_batch_gains))

        #   randomly pick a mini batch from replay buffer
        mb_indices = [i for i in range(num_steps)]
        np.random.shuffle(mb_indices)
        num_mbatches = math.ceil(num_steps / mini_batch)
        # train n_epochs for each batch
        for _ in range(n_epochs):
            for b in range(num_mbatches):
                # pick a mini batch
                # print(f'train with mini-batch {b}')
                batch = mb_indices[b * mini_batch : (b + 1) * mini_batch]

                #   calculate the probability of action under new policy
                #    - by passing the observation through the actor-critic network
                a = actions[batch]
                _, a_logprob, a_entropy, v = agent.get_action_value(obs[batch], a)

                #   calculate the portion of probabilities of action under new and old policy
                p_theta_log = a_logprob - action_probs[batch]
                p_theta = torch.exp(p_theta_log)

                #   calculate the clipped policy surrogate objective function
                adv = advantages[batch]
                loss_cp1 = - adv * p_theta
                loss_cp2 = - adv * torch.clamp(p_theta, 1 - clip_coef, 1 + clip_coef)

                #   calculate the policy gradient loss ( multiple clipped surrogate fuction with GAE )
                pg_loss = torch.max(loss_cp1, loss_cp2).mean()

                #   calculate the value loss
                g = gains[batch]
                v = v.view(g.shape)
                v_loss = ((v - g) ** 2).mean() / 2

                #   entropy loss
                entropy_loss = a_entropy.mean()

                #   calculate the actor-critic objective function
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                # step and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return agent, total_gains

def evaluate(agent: nn.Module, hype_params, isTuning):
    env_id = hype_params["env_id"]
    # gamma = hype_params["gamma"]
    evaluation = INIT_GAIN
    # evaluate the model
    eval_max_steps = hype_params["num_steps"]
    eval_rounds = 10
    eval_env = gym.make(env_id, render_mode='rgb_array')
    gains = []
    frames_list = []
    rewards = []
    agent.eval()
    for _ in range(eval_rounds):
        # print(f'evaluate round {round}')
        gain = 0
        s, _ = eval_env.reset()
        for t in range(eval_max_steps):
            if not isTuning:
                frames_list.append(eval_env.render())
            with torch.no_grad():
                a, _, _, _ = agent.get_action_value(torch.Tensor(s).to(device))
            s, r, done, _, _ = eval_env.step(a.cpu().numpy())
            # gain = r + gain * gamma
            gain += r # for lunar lander, the score is accumulated without decaying
            rewards.append('{:.2f}'.format(r))
            if done:
                break

        gains.append(gain)
    t_gains = torch.Tensor(gains)
    evaluation = t_gains.mean()
    deviation = t_gains.std()

    return evaluation, deviation, frames_list

def recordVideo(filename, frames):
    # visualize
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (600, 400))
    for frame in frames:
        frame = np.uint8(frame)
        video.write(frame)
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimze-hype-params", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    args = parser.parse_args()
    final_params = None
    if args.optimze_hype_params:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        
        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        final_params = trial.params
    else:
        final_params = {}
        final_params["num_steps"] = 512
        final_params["n_envs"] = 4
        final_params["gamma"] = 0.993 # 0.993, 0.995
        final_params["mini_batch"] = 8 # 6
        final_params["n_epochs"] = 8
        final_params["learning_rate"] = 1e-3
        final_params["clip_coef"] = 0.1
        final_params["ent_coef"] = 0.05 # 0.1, 0.05, 0.03
        final_params["vf_coef"] = 0.4

    print("  Params: ")
    for key, value in final_params.items():
        print("    {}: {}".format(key, value))

    print("Start training with optimzed hype parameters ... ")
    if final_params != None:
        final_params["env_id"] = ENV_ID
        # final_params["evaluation"] = INIT_GAIN
        agent, _ = ppo(final_params, False)
        print("Evaluating model ... ")
        evaluation, deviation, frames = evaluate(agent, final_params, False)
        print("Final average gain {} +/- {}".format(evaluation, deviation))

        path = os.path.realpath(__file__)
        dir = os.path.dirname(path)
        recordVideo(dir + os.path.sep + ENV_ID + '.mp4', frames)
