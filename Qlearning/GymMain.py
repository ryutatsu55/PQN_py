import gymnasium as gym
import numpy as np
from DQNAgent import DQNAgent

# --- ハイパーパラメータ ---
T_sample = 0.001  # [s] 高速な状態サンプリング周期 (例: 1ms)
T_action = 0.1    # [s] 低速な行動選択周期 (例: 100ms)
action_interval = int(T_action / T_sample) # => 100ステップ

# --- 初期化 ---
env = gym.make('CartPole-v1', render_mode='human') # CartPoleは行動しないと状態が変化する
agent = DQNAgent(...) # 内部にSNNシミュレータと読み出し層を持つ

for episode in range(num_episodes):
    obs, info = env.reset()
    agent.reservoir.reset() # SNNリザバーの内部状態をリセット
    
    # 最初の行動を決定するための初期観測
    for _ in range(action_interval):
        encoded_spikes = agent.reservoir._encode_single_state(obs, 1) # 1ステップ分のスパイク
        agent.reservoir.step(encoded_spikes[0])
    
    reservoir_state_r = agent.reservoir.get_readout_state(...)
    action = agent.get_action(reservoir_state_r) # ε-greedyなどで最初の行動を選択

    done = False
    while not done:
        
        rewards_in_interval = []
        
        # --- 低速ループ (1回の行動選択) ---
        
        # 1. 前回の行動選択時のリザバー状態を保存
        prev_reservoir_state_r = agent.reservoir.get_readout_state(...)

        # 2. 高速ループ (次の行動選択までの状態サンプリング)
        for _ in range(action_interval):
            # 決定された行動 `action` を環境に入力し、1ステップ進める
            # (注: 環境のステップとT_sampleを一致させる必要がある)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards_in_interval.append(reward)

            # SNNリザバーにリアルタイムで観測を入力
            encoded_spikes = agent.reservoir._encode_single_state(next_obs, 1)
            agent.reservoir.step(encoded_spikes[0])

            if done:
                break
        
        # 3. 低速ループの最後に、次の行動を選択
        current_reservoir_state_r = agent.reservoir.get_readout_state(...)
        next_action = agent.get_action(current_reservoir_state_r) # 次のインターバルで使う行動
        
        # 4. 報酬を平均化
        avg_reward = np.mean(rewards_in_interval)
        
        # 5. リプレイバッファに保存
        #    保存するのは、観測o_tではなく、リザバーの状態r_t
        agent.replay_buffer.add(prev_reservoir_state_r, action, avg_reward, current_reservoir_state_r, done)

        action = next_action

        if done:
            break
            
        # 6. 学習 (この部分は変更なし)
        agent.update()