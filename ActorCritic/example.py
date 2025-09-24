import numpy as np
from ActorCritic.agent import Agent
import gymnasium as gym

# env = SatelliteBroadbandEnv(dataset_path = "optim_train_set.csv")
env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")  # default goal_velocity=0
print(env.action_space.shape[0])
agent = Agent(alpha = 1e-3, beta = 1e-3, input_dims= env.observation_space.shape[0], env= env,
              gamma = 0.5, n_actions = env.action_space.shape[0])

episodes = 50
scores = []
for episode in range(episodes) :
    (current_state, info) = env.reset()
    score = 0
    done = False
    while not done :
        action = agent.choose_action(current_state)
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        agent.remember(current_state, action, reward, next_state, done)
        agent.learn()
        current_state = next_state
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print(f'Episode: {episode}, Score: {score:.2f}, Average Score: {avg_score:.2f}')
