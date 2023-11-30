import gymnasium as gym

'''
Action Space: Discrete, Continuous
Continuous: [steering, gas, breaking]
steering: 0 straignt, 1 is full right, -1 full left 
Discrete:
    0: do nothing
    1: steer left 
    2: steer rigth 
    3: gas 
    4: brake 
Observation Space: RGB image, 96px by 96px 
Reward: -0.1 for every frame, +1k/N for every track tile visited where 
N is the number of track tiles visited in the track  1k-0.1*f = R points 
f is the number of frames
GOAL: 900 POINTS
'''

env = gym.make('CarRacing-v2',
            render_mode="human", 
            domain_randomize=True, 
            continuous=False)

observation, info = env.reset()
for _ in range(1_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
