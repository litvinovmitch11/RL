import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_plot_qlearning():
    """Load and plot Q-learning results"""
    print("Analyzing Q-learning results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Custom Q-learning
    custom_dir = "outputs/4x4/ql-custom"
    if os.path.exists(custom_dir):
        # Load all run statistics
        custom_rewards = []
        for run_file in glob(f"{custom_dir}/run*_stats.npz"):
            data = np.load(run_file)
            custom_rewards.append(data['episode_rewards'])
        
        if custom_rewards:
            # Average across runs
            max_len = max(len(r) for r in custom_rewards)
            padded = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=np.nan) 
                     for r in custom_rewards]
            custom_avg = np.nanmean(padded, axis=0)
            custom_std = np.nanstd(padded, axis=0)
            
            # Plot rewards
            ax = axes[0]
            episodes = np.arange(len(custom_avg))
            ax.plot(episodes, custom_avg, label='Custom Q-learning', linewidth=2)
            ax.fill_between(episodes, custom_avg - custom_std, custom_avg + custom_std, alpha=0.3)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Reward')
            ax.set_title('Custom Q-learning Performance')
            ax.legend()
            ax.grid(True)
    
    # Baseline Q-learning
    baseline_files = glob("outputs/4x4/baseline/ql-baseline*.csv")
    if baseline_files:
        # Load and process CSV files
        baseline_rewards = []
        for file in baseline_files:
            df = pd.read_csv(file)
            if 'system_total_reward' in df.columns:
                # Sum rewards per episode
                episode_reward = df['system_total_reward'].sum()
                baseline_rewards.append(episode_reward)
        
        if baseline_rewards:
            ax = axes[1]
            ax.bar(['Baseline Q-learning'], [np.mean(baseline_rewards)], 
                   yerr=[np.std(baseline_rewards)], capsize=10)
            ax.set_ylabel('Average Total Reward')
            ax.set_title('Baseline Q-learning Performance')
            ax.grid(True, axis='y')
    
    # Comparison
    ax = axes[2]
    if 'custom_avg' in locals() and baseline_rewards:
        comparison_data = {
            'Custom': np.mean(custom_avg),
            'Baseline': np.mean(baseline_rewards)
        }
        ax.bar(comparison_data.keys(), comparison_data.values())
        ax.set_ylabel('Average Reward')
        ax.set_title('Q-learning: Custom vs Baseline')
        ax.grid(True, axis='y')
    
    # Epsilon decay
    ax = axes[3]
    if os.path.exists(custom_dir):
        for run_file in glob(f"{custom_dir}/run*_stats.npz"):
            data = np.load(run_file)
            ax.plot(data['epsilon_values'], alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Epsilon Decay (Custom Q-learning)')
        ax.grid(True)
    
    plt.suptitle('Q-learning Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/qlearning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_and_plot_dqn():
    """Load and plot DQN results"""
    print("Analyzing DQN results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Custom DQN
    custom_dir = "outputs/big-intersection/dqn-custom"
    if os.path.exists(f"{custom_dir}/training_stats.npz"):
        data = np.load(f"{custom_dir}/training_stats.npz")
        
        # Plot rewards
        ax = axes[0]
        episodes = np.arange(len(data['episode_rewards']))
        ax.plot(episodes, data['episode_rewards'], label='Custom DQN', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Custom DQN Performance')
        ax.legend()
        ax.grid(True)
        
        # Plot loss
        ax = axes[1]
        ax.plot(episodes, data['episode_losses'], label='Training Loss', color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Loss')
        ax.set_title('Custom DQN Training Loss')
        ax.legend()
        ax.grid(True)
        
        # Plot epsilon
        ax = axes[2]
        ax.plot(episodes, data['epsilon_values'], label='Epsilon', color='green', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate Decay')
        ax.legend()
        ax.grid(True)
    
    # Baseline DQN analysis would go here
    # For now, just show custom results
    
    plt.suptitle('DQN Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/dqn_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_traffic_metrics():
    """Compare traffic metrics from CSV files"""
    print("Comparing traffic metrics...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = [
        ('system_total_waiting_time', 'Total Waiting Time (s)'),
        ('system_total_stopped', 'Total Stopped Vehicles'),
        ('system_mean_waiting_time', 'Mean Waiting Time (s)'),
        ('system_mean_speed', 'Mean Speed (m/s)'),
        ('system_total_reward', 'Total Reward'),
        ('system_avg_queue', 'Average Queue Length')
    ]
    
    # Collect data from all implementations
    implementations = {
        'Custom DQN': 'outputs/big-intersection/dqn-custom/traffic*.csv',
        'Custom QL': 'outputs/4x4/ql-custom/run*.csv',
        'Baseline DQN': 'outputs/big-intersection/baseline/dqn-baseline*.csv',
        'Baseline QL': 'outputs/4x4/baseline/ql-baseline*.csv'
    }
    
    for idx, (metric, title) in enumerate(metrics):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        metric_values = {}
        
        for impl_name, pattern in implementations.items():
            files = glob(pattern)
            if files:
                all_values = []
                for file in files[:3]:  # Limit to first 3 files
                    try:
                        df = pd.read_csv(file)
                        if metric in df.columns:
                            all_values.append(df[metric].mean())
                    except:
                        continue
                
                if all_values:
                    metric_values[impl_name] = all_values
        
        if metric_values:
            # Create box plot
            labels = []
            data = []
            for impl, values in metric_values.items():
                labels.append(impl)
                data.append(values)
            
            ax.boxplot(data, labels=labels)
            ax.set_ylabel(title)
            ax.set_title(f'{title} Comparison')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y')
    
    plt.suptitle('Traffic Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/traffic_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report():
    """Create a summary report of all results"""
    print("Creating summary report...")
    
    summary = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'implementations': {},
        'comparisons': {}
    }
    
    # Q-learning comparison
    custom_dir = "outputs/4x4/ql-custom"
    if os.path.exists(custom_dir):
        custom_rewards = []
        for run_file in glob(f"{custom_dir}/run*_stats.npz"):
            data = np.load(run_file)
            custom_rewards.extend(data['episode_rewards'])
        
        if custom_rewards:
            summary['implementations']['Custom Q-learning'] = {
                'average_reward': float(np.mean(custom_rewards)),
                'std_reward': float(np.std(custom_rewards)),
                'max_reward': float(np.max(custom_rewards)),
                'min_reward': float(np.min(custom_rewards)),
                'num_episodes': len(custom_rewards)
            }
    
    # DQN comparison
    custom_dir = "outputs/big-intersection/dqn-custom"
    if os.path.exists(f"{custom_dir}/training_stats.npz"):
        data = np.load(f"{custom_dir}/training_stats.npz")
        summary['implementations']['Custom DQN'] = {
            'average_reward': float(np.mean(data['episode_rewards'])),
            'std_reward': float(np.std(data['episode_rewards'])),
            'max_reward': float(np.max(data['episode_rewards'])),
            'min_reward': float(np.min(data['episode_rewards'])),
            'num_episodes': len(data['episode_rewards']),
            'final_epsilon': float(data['epsilon_values'][-1]),
            'average_loss': float(np.mean(data['episode_losses']))
        }
    
    # Save summary
    with open('outputs/summary_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    for impl, stats in summary['implementations'].items():
        print(f"\n{impl}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print(f"Report saved to: outputs/summary_report.json")
    print("=" * 60)

if __name__ == "__main__":
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    print("=" * 60)
    print("ANALYZING RESULTS")
    print("=" * 60)
    
    # Load and plot results
    load_and_plot_qlearning()
    load_and_plot_dqn()
    compare_traffic_metrics()
    create_summary_report()
    
    print("\nAnalysis complete!")
    print("Plots saved to outputs/ directory")
