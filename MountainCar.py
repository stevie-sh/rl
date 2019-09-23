import gym
import numpy as np

# https://github.com/openai/gym/wiki/MountainCar-v0

env = gym.make("MountainCar-v0")
env.reset()


### Box: [position, velocity]
print('env high', env.observation_space.high) # high value of observation space
print('env low', env.observation_space.low) # low value of observation space
print(env.action_space.n) # number of actions

done = False

### Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 500
EPSILON = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

### Discretize space - bucketing the observations
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) 
DISCRETE_OS_WINDOW_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Initialize q
q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n])) # action x state matrix

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_OS_WINDOW_SIZE
    return tuple(discrete_state.astype(np.int))


#print(DISCRETE_OS_WINDOW_SIZE)
#print(q_table)

print('shape of table is', np.shape(q_table))
for episode in range(30):
    render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    print(f"episode {episode}")

    if episode % SHOW_EVERY == 0 or episode == 0:
        print(q_table)

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) * DISCOUNT
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0] >= env.goal_position:
            print('we made it', episode)
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
        
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= EPSILON_DECAY_VALUE


env.close()
