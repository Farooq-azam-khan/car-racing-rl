import gymnasium as gym
from stable_baselines3 import PPO 

env = gym.make('CarRacing-v2',
            render_mode="human",
            domain_randomize=True,
            continuous=False) 
observation, info = env.reset(options={'randomize': False})
print(env.action_space)
model = PPO.load('rl_models/car_ppo_v1', env=env)


for _ in range(1_000):
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated: 
        print('DONE:', reward) 
        env.close() 
env.close() 

