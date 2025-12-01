import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ImprovedDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (np.array(states, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=bool),
                indices,
                np.array(weights, dtype=np.float32))
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update=1000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.policy_net = ImprovedDQN(state_dim, action_dim).to(device)
        self.target_net = ImprovedDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.steps_done = 0
        self.losses = []
        
    def select_action(self, state):
        self.steps_done += 1
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (~dones)
        
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        
        self.memory.update_priorities(indices, td_errors)
        
        loss = (weights * (current_q - target_q.detach()).pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()


if __name__ == "__main__":
    config = {
        'total_steps': 20000,  # Reduced for faster training
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.9995,
        'memory_size': 10000,
        'batch_size': 32,
        'target_update': 1000,
        'num_seconds': 1800,  # Reduced episode length
        'min_green': 5,
        'max_green': 60,
        'yellow_time': 4,
        'delta_time': 5,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = "outputs/big-intersection/dqn-custom"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CUSTOM DQN IMPLEMENTATION WITH IMPROVEMENTS")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    
    # Environment setup
    env = SumoEnvironment(
        net_file="../big-intersection/big-intersection.net.xml",
        route_file="../big-intersection/routes.rou.xml",
        out_csv_name=f"{output_dir}/traffic",
        single_agent=True,
        use_gui=False,
        num_seconds=config['num_seconds'],
        yellow_time=config['yellow_time'],
        min_green=config['min_green'],
        max_green=config['max_green'],
        delta_time=config['delta_time'],
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Configuration: {config}")
    print("-" * 60)
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        target_update=config['target_update']
    )
    
    stats = {
        'episode_rewards': [],
        'episode_steps': [],
        'episode_losses': [],
        'epsilon_values': [],
        'episode_times': []
    }
    
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    episode_reward = 0
    episode_steps = 0
    episode_loss = 0
    episode_count = 0
    episode_start_time = time.time()
    
    print("Starting training...\n")
    
    while agent.steps_done < config['total_steps']:
        action = agent.select_action(state)
        
        step_result = env.step(action)
        
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result
        
        agent.store_transition(state, action, reward, next_state, done)
        
        loss = agent.learn()
        if loss > 0:
            episode_loss += loss
        
        episode_reward += reward
        episode_steps += 1
        state = next_state
        
        if done or episode_steps >= config['num_seconds'] // config['delta_time']:
            episode_time = time.time() - episode_start_time
            
            stats['episode_rewards'].append(episode_reward)
            stats['episode_steps'].append(episode_steps)
            stats['episode_losses'].append(episode_loss / max(1, episode_steps))
            stats['epsilon_values'].append(agent.epsilon)
            stats['episode_times'].append(episode_time)
            
            print(f"Episode {episode_count:3d} | "
                  f"Steps: {episode_steps:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {episode_loss/max(1, episode_steps):.4f} | "
                  f"Time: {episode_time:.1f}s | "
                  f"Total: {agent.steps_done:5d}/{config['total_steps']}")
            
            env.save_csv(f"{output_dir}/traffic", episode_count)
            
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0
            episode_steps = 0
            episode_loss = 0
            episode_count += 1
            episode_start_time = time.time()
    
    torch.save(agent.policy_net.state_dict(), f"{output_dir}/dqn_model_final.pth")
    np.savez(f"{output_dir}/training_stats.npz", **stats)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {agent.steps_done}")
    print(f"Average reward: {np.mean(stats['episode_rewards']):.2f}")
    print(f"Max reward: {np.max(stats['episode_rewards']):.2f}")
    print(f"Final epsilon: {stats['epsilon_values'][-1]:.3f}")
    print(f"Results saved to: {output_dir}")
    
    env.close()
