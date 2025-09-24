import numpy as np
import torch
import gym
from policy import Policy
import numpy as np

class SimulatedAnnealingPolicyOptimizer:
    def __init__(self, env, temp=150, cooling_rate=0.99, min_temp=1, max_episodes=900):
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n  # Fixed typo: was `action_sim`
        self.temp = temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_episodes = max_episodes
        self.policy = Policy(self.input_dim, self.action_dim)

    def evaluate_policy(self, policy, episodes=10):
        """Evaluates the total reward of a given policy."""
        total_reward = 0
        for _ in range(episodes):
            current_state = self.env.reset()[0]
            done = False
            while not done:
                action = torch.argmax(policy.forward(current_state)).item()
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                current_state = next_state
        return total_reward / episodes  # Average reward

    def perturb_policy(self, policy):
        """Creates a slightly modified version of the policy."""
        new_policy = Policy(self.input_dim, self.action_dim)  # New instance
        new_policy.load_state_dict(policy.state_dict())  # Copy weights

        # Add small noise to parameters
        with torch.no_grad():
            for param in new_policy.parameters():
                param += torch.randn_like(param) * 0.1

        return new_policy

    def run(self):
        """Optimizes the policy parameters using Simulated Annealing."""
        best_policy = self.policy
        best_reward = self.evaluate_policy(best_policy)

        for episode in range(self.max_episodes):
            # Generate a new candidate policy by modifying parameters slightly
            new_policy = self.perturb_policy(best_policy)
            new_reward = self.evaluate_policy(new_policy)

            # Accept if it's better, or with probability if worse
            delta = new_reward - best_reward
            if delta > 0 or np.exp(delta / self.temp) > np.random.random():
                best_policy = new_policy
                best_reward = new_reward

            # Reduce temperature
            self.temp *= self.cooling_rate
            print(f"episode {episode} : policy performance {best_reward}")
            if self.temp < self.min_temp:
                break  # Stop if temperature is too low

        return best_policy, best_reward

# Run SA on a discrete RL environment
env = gym.make("CartPole-v1")  # Changed to CartPole (FrozenLake needs tabular methods)
agent = SimulatedAnnealingPolicyOptimizer(env)
optimal_policy, policy_performance = agent.run()

print("Optimal Policy performance :")
print(policy_performance)
