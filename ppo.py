import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('CarRacing-v2',
            render_mode=None, # "human",
            domain_randomize=True,
            continuous=False)
observation, info = env.reset(options={'randomize': False})

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(2e4), progress_bar=True)
model.save('rl_models/car_ppo_v1')

