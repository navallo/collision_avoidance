import gym
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2

import collision_avoidance

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('collision_avoidance-v0') for i in range(n_cpu)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

timestr = time.strftime("%Y%m%d_%H%M%S")
path = timestr+'_ppo2_collision_avoidance'

model.save(path)

# del model # remove to demonstrate saving and loading
# model = PPO2.load(path)

# Enjoy trained agent
env = gym.make('collision_avoidance-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()