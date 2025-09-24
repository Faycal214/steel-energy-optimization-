import numpy as np
import gymnasium as gym
from agent import dqn

if __name__ == '__main__' :
    # Initialize the environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Initialize the parameters
    num_episodes = 1000
    learning_rate = 0.0001
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    memory_size = 1000
    batch_size = 64
    combined = False
    
    # Initialize the DQN agent
    agent = dqn(state_dim = env.observation_space.shape[0], action_dim = env.action_space.n, 
                learning_rate = learning_rate, gamma = 0.99, epsilon = epsilon, batch_size = batch_size, 
                max_memory_size = memory_size, combined = combined)
    
    scores = []
    for episode in range(num_episodes) :
        score = 0
        (current_state, _) = env.reset()
        done = False
        while not done :
            action = agent.choose_action(current_state)
            next_state, reward, done, _ , _= env.step(action)
            score += reward
            agent.memory.store_transition(current_state, action, reward, next_state, done)
            agent.learn()
            current_state = next_state
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"episode {episode} | score {score} | avg score {avg_score} | eps {agent.epsilon}")
