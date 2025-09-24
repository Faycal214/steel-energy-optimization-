import numpy as np
from networks import PPOAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
N = 250
agent = PPOAgent(env.observation_space.shape[0], n_actions = env.action_space.n)
n_games = 950

best_score = 0
score_history = []

learn_iter = 0
avg_score = 0
n_steps = 0

for i in range(n_games) :
    (current_state, _) = env.reset()
    done = False
    score = 0
    while not done :
        action, prob, val = agent.choose_action(current_state)
        next_state, reward, done, _, _ = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(current_state, action, prob, val, reward, done)
        if n_steps % N == 0 :
            agent.learn()
            learn_iter += 1
        current_state = next_state
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    
    if avg_score > best_score :
        best_score = avg_score 
    
    print(f"episode {i}, score {score}, avg_score {avg_score}")
