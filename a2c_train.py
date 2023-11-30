import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make('CarRacing-v2', render_mode=None,  domain_randomize=True,
               continuous=False)
observation, info = env.reset(options={'randomize': False})

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(2e5), progress_bar=True)
model.save('rl_models/car_ppo_v1')

