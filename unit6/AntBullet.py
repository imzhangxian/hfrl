import pybullet_envs
import panda_gym
import gym

import os

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login

# create the environment and model
train_steps=1000000

env_id="AntBulletEnv-v0" 
env = make_vec_env(lambda: gym.make(env_id), n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = A2C(
    policy="MlpPolicy",
    env=env,
    gae_lambda=0.9,
    gamma=0.99,
    learning_rate=0.00096,
    max_grad_norm=0.5,
    n_steps=8,
    vf_coef=0.4,
    ent_coef=0.0,
    policy_kwargs=dict(log_std_init=-2, ortho_init=False),
    normalize_advantage=False,
    use_rms_prop=True,
    use_sde=True,
    verbose=1,
)

# train the model
model.learn(train_steps)

# Save the model and  VecNormalize statistics when saving the agent
model.save("a2c-AntBulletEnv-v0")
env.save("vec_normalize.pkl")

# save and evaluate the model
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make(env_id)])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# Load the agent
model = A2C.load("a2c-AntBulletEnv-v0")

mean_reward, std_reward = evaluate_policy(model, env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
