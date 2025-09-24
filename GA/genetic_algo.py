import numpy as np
import random
import torch
from genetic_agent import agent_policy

class GeneticAgent :
    def __init__(self, env, population_size, nb_generations, mutation_rate) :
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.population_size = population_size
        self.nb_generations = nb_generations
        self.mutation_rate = mutation_rate
        self.initial_population = [self.create_agent() for _ in range(population_size)]
    
    def create_agent(self) :
        agent = agent_policy(self.state_space, self.action_space)
        return agent
    
    def fitness(self, individual, episodes = 5) :
        total_rewards = 0
        for _ in range(episodes) :
            (current_state, _) = self.env.reset()
            done = False
            while not done :
                action = torch.argmax(individual.forward(current_state)).item()
                (next_state, reward, done, _, _) = self.env.step(action)
                total_rewards += reward
                current_state = next_state
        return total_rewards / episodes
    
    def croos_over(self, parent1, parent2) :
        child = self.create_agent()
        
        crossover_point = random.randint(1, self.state_space)
        
        child.fc1.weight.data = torch.cat((parent1.fc1.weight.data[:, :crossover_point], parent2.fc1.weight.data[:, crossover_point:]), dim=1)
        child.fc1.bias.data = torch.cat((parent1.fc1.bias.data[:crossover_point], parent2.fc1.bias.data[crossover_point:]), dim=0)
        
        child.fc2.weight.data = torch.cat((parent1.fc2.weight.data[:, :crossover_point], parent2.fc2.weight.data[:, crossover_point:]), dim=1)
        child.fc2.bias.data = torch.cat((parent1.fc2.bias.data[:crossover_point], parent2.fc2.bias.data[crossover_point:]), dim=0)
        
        child.fc3.weight.data = torch.cat((parent1.fc3.weight.data[:, :crossover_point], parent2.fc3.weight.data[:, crossover_point:]), dim=1)
        child.fc3.bias.data = torch.cat((parent1.fc3.bias.data[:crossover_point], parent2.fc3.bias.data[crossover_point:]), dim=0)
        
        return child
    
    def create_generation(self, old_population) :
        new_population = []
        
        for _ in range(self.population_size) :
            parent_index_1, parent_index_2 = random.sample(range(self.population_size), k= 2)
            parent1 = old_population[parent_index_1]
            parent2 = old_population[parent_index_2]
            
            child = self.croos_over(parent1, parent2)
            new_population.extend([parent1, parent2, child])
        
        new_population = self.selection(new_population)
        
        return new_population[:self.population_size]
    
    def selection(self, population) :
        fitness_scores = np.array([self.fitness(individual) for individual in population])
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        # fitness = [fitness[i] for i in sorted_indices]
        return population
    
    def mutation(self, population) :
        for i, individual in enumerate(population) :
            if random.random() < self.mutation_rate :
                mutation_point = random.randint(0, self.state_space - 1)
                individual.fc1.weight.data[mutation_point] += random.uniform(-0.5, 0.5)
                individual.fc1.bias.data[mutation_point] += random.uniform(-0.5, 0.5)
                
                mutation_point = random.randint(0, self.state_space - 1)
                individual.fc2.weight.data[mutation_point] += random.uniform(-0.5, 0.5)
                individual.fc2.bias.data[mutation_point] += random.uniform(-0.5, 0.5)
                
                mutation_point = random.randint(0, self.action_space - 1)
                individual.fc3.weight.data[mutation_point] += random.uniform(-0.5, 0.5)
                individual.fc3.bias.data[mutation_point] += random.uniform(-0.5, 0.5)
                
                population[i] = individual
        return population
    
    def run(self) :
        best_policies_performance = {}
        for generation in range(self.nb_generations) :
            if generation == 0 :
                population = self.initial_population
            
            new_generation = self.create_generation(population)
            new_generation = self.mutation(new_generation)
            
            new_generation = self.selection(new_generation)
            
            best_policy = new_generation[0]
            best_policy_performance = self.fitness(best_policy)
            # Store the best policy information for this generation
            best_policies_performance[generation] = [generation, best_policy, best_policy_performance]
            print(f"generation {generation} : best policy perforrmance {best_policy_performance}")


            population = new_generation
        
        return best_policies_performance[0]
            
            