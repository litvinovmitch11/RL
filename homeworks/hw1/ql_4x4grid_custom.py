import os
import sys
import numpy as np
import random
from collections import defaultdict
import time

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

class CustomQLAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.05, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        
        self.learning_steps = 0
        self.total_reward = 0
        
    def _state_to_key(self, state):
        if isinstance(state, np.ndarray):
            state = tuple(np.round(state, 2))
        return state
    
    def act(self, state):
        state_key = self._state_to_key(state)
        
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state_key][action] = new_q
        
        self.learning_steps += 1
        self.total_reward += reward
        
        if self.learning_steps % 100 == 0:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return new_q - current_q


if __name__ == "__main__":
    # Parameters (similar to original)
    alpha = 0.1
    gamma = 0.99
    decay = 0.995
    runs = 3  # Reduced for speed
    episodes = 20  # Reduced for speed
    
    output_dir = "outputs/4x4/ql-custom"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CUSTOM Q-LEARNING IMPLEMENTATION")
    print("=" * 60)
    
    for run in range(1, runs + 1):
        print(f"\n--- Run {run}/{runs} ---")
        
        # Environment setup
        env = SumoEnvironment(
            net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
            route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
            out_csv_name=f"{output_dir}/run{run}",
            use_gui=False,
            num_seconds=1000,  # Reduced for faster training
            min_green=5,
            delta_time=5,
        )
        
        initial_states = env.reset()
        ql_agents = {
            ts: CustomQLAgent(
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                epsilon=0.1,
                epsilon_decay=decay,
                min_epsilon=0.01
            )
            for ts in env.ts_ids
        }
        
        run_stats = {
            'episode_rewards': [],
            'episode_steps': [],
            'epsilon_values': [],
            'q_table_sizes': []
        }
        
        for episode in range(1, episodes + 1):
            episode_start = time.time()
            
            if episode != 1:
                initial_states = env.reset()
            
            states = initial_states
            done = {"__all__": False}
            episode_rewards = {ts: 0 for ts in env.ts_ids}
            episode_steps = 0
            
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act(states[ts]) for ts in ql_agents.keys()}
                
                next_states, rewards, done, _ = env.step(action=actions)
                
                for ts in ql_agents.keys():
                    td_error = ql_agents[ts].learn(
                        state=states[ts],
                        action=actions[ts],
                        reward=rewards[ts],
                        next_state=next_states[ts],
                        done=done[ts]
                    )
                    episode_rewards[ts] += rewards[ts]
                
                states = next_states
                episode_steps += 1
            
            episode_time = time.time() - episode_start
            avg_reward = np.mean(list(episode_rewards.values()))
            avg_epsilon = np.mean([agent.epsilon for agent in ql_agents.values()])
            avg_q_size = np.mean([len(agent.q_table) for agent in ql_agents.values()])
            
            run_stats['episode_rewards'].append(avg_reward)
            run_stats['episode_steps'].append(episode_steps)
            run_stats['epsilon_values'].append(avg_epsilon)
            run_stats['q_table_sizes'].append(avg_q_size)
            
            print(f"Episode {episode:3d}/{episodes} | "
                  f"Reward: {avg_reward:8.2f} | "
                  f"Steps: {episode_steps:4d} | "
                  f"Epsilon: {avg_epsilon:.3f} | "
                  f"Q-table: {avg_q_size:4.0f} | "
                  f"Time: {episode_time:.1f}s")
            
            env.save_csv(f"{output_dir}/run{run}", episode)
        
        np.savez(f"{output_dir}/run{run}_stats.npz", **run_stats)
        
        print(f"\nRun {run} Summary:")
        print(f"  Average reward: {np.mean(run_stats['episode_rewards']):.2f}")
        print(f"  Max reward: {np.max(run_stats['episode_rewards']):.2f}")
        print(f"  Final epsilon: {run_stats['epsilon_values'][-1]:.3f}")
        print(f"  Average Q-table size: {np.mean(run_stats['q_table_sizes']):.0f}")
        
        env.close()
    
    print("\n" + "=" * 60)
    print("CUSTOM Q-LEARNING COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
