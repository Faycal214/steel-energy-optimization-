import numpy as np
import gymnasium as gym
from genetic_algo import GeneticAgent

# creating the environment
env = gym.make("CartPole-v1")
population_size = 200
nb_generations = 100
mutation_rate = 0.9
agent = GeneticAgent(env= env, population_size = population_size, nb_generations = nb_generations, mutation_rate = mutation_rate)

# training the agent
episode, policy, policy_performance = agent.run()