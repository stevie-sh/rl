import gym
import time
env = gym.make('Breakout-v0')
env.reset()

for _ in range (5):
    observation = env.reset()
    for t in range(1000):
        env.render()
        time.sleep(0.1)
        # Take a single step
        action = env.action_space.sample() 
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
