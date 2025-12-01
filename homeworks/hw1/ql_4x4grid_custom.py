import os
import sys
import time

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

if __name__ == "__main__":
    # Параметры как у кастомной реализации
    alpha = 0.1
    gamma = 0.99
    decay = 0.995  # Изменено с 1 на 0.995 для затухания epsilon
    runs = 3  # Уменьшено с 30 для скорости
    episodes = 20  # Увеличено с 4 для лучшего обучения
    
    # Создаем директорию для выходных данных
    output_dir = "outputs/4x4/ql"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("BASELINE Q-LEARNING IMPLEMENTATION")
    print("(With parameters matching custom implementation)")
    print("=" * 60)
    print(f"Runs: {runs}")
    print(f"Episodes per run: {episodes}")
    print(f"Alpha: {alpha}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon decay: {decay}")
    print("=" * 60)

    # Создаем среду с теми же параметрами, что у кастомной
    env = SumoEnvironment(
        net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        out_csv_name=f"{output_dir}/baseline_run",  # Базовое имя для CSV
        use_gui=False,
        num_seconds=1000,  # Такое же как у кастомной (уменьшено с 80000)
        min_green=5,
        delta_time=5,
    )

    for run in range(1, runs + 1):
        print(f"\n--- Run {run}/{runs} ---")
        run_start_time = time.time()
        
        # Собираем статистику для этого запуска
        run_stats = {
            'episode_rewards': [],
            'episode_steps': [],
            'episode_times': [],
            'epsilon_values': []  # Будем отслеживать epsilon
        }
        
        initial_states = env.reset()
        
        # Создаем агентов с параметрами как у кастомной реализации
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                # Используем те же параметры epsilon что у кастомной
                exploration_strategy=EpsilonGreedy(
                    initial_epsilon=0.1,  # Как у кастомной
                    min_epsilon=0.01,     # Как у кастомной
                    decay=decay           # Как у кастомной
                ),
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            episode_start_time = time.time()
            
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            episode_steps = 0
            episode_reward = 0
            
            while not done["__all__"]:
                # Выбираем действия
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                # Выполняем шаг
                s, r, done, info = env.step(action=actions)

                # Обучаем агентов
                for agent_id in s.keys():
                    ql_agents[agent_id].learn(
                        next_state=env.encode(s[agent_id], agent_id), 
                        reward=r[agent_id]
                    )
                    episode_reward += r[agent_id]
                
                episode_steps += 1
            
            # Эпизод завершен
            episode_time = time.time() - episode_start_time
            
            # Получаем текущее значение epsilon (примерное, берем у первого агента)
            if ql_agents:
                first_agent = list(ql_agents.values())[0]
                current_epsilon = first_agent.exploration_strategy.epsilon
            else:
                current_epsilon = 0.1
            
            # Сохраняем статистику
            run_stats['episode_rewards'].append(episode_reward)
            run_stats['episode_steps'].append(episode_steps)
            run_stats['episode_times'].append(episode_time)
            run_stats['epsilon_values'].append(current_epsilon)
            
            # Выводим прогресс
            print(f"Episode {episode:3d}/{episodes} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Steps: {episode_steps:4d} | "
                  f"Epsilon: {current_epsilon:.3f} | "
                  f"Time: {episode_time:.1f}s")
            
            # Сохраняем CSV для этого эпизода
            # Используем ту же схему именования что у кастомной
            env.save_csv(f"{output_dir}/baseline_run{run}", episode)
        
        # Запуск завершен
        run_time = time.time() - run_start_time
        
        # Сохраняем статистику запуска в файл (как у кастомной)
        import numpy as np
        np.savez(f"{output_dir}/baseline_run{run}_stats.npz", **run_stats)
        
        # Выводим сводку по запуску
        print(f"\nRun {run} Summary:")
        print(f"  Average reward: {np.mean(run_stats['episode_rewards']):.2f}")
        print(f"  Max reward: {np.max(run_stats['episode_rewards']):.2f}")
        print(f"  Min reward: {np.min(run_stats['episode_rewards']):.2f}")
        print(f"  Final epsilon: {run_stats['epsilon_values'][-1]:.3f}")
        print(f"  Average steps per episode: {np.mean(run_stats['episode_steps']):.0f}")
        print(f"  Total run time: {run_time:.1f}s")

    env.close()
    
    print("\n" + "=" * 60)
    print("BASELINE Q-LEARNING COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
