class DQNAgent:
    def __init__(self, ...):
        # ...
        self.q_network = ReservoirQNetwork(...) # 内部にSNNシミュレータと読み出し層を持つ
        self.replay_buffer = ReplayBuffer(...)
        self.sequence_length = 4 # 学習時に遡る履歴の長さ

    def get_action(self, state, is_training=True):
        """
        【リアルタイム処理】
        現在の観測(state)を受け取り、継続中のシミュレーションを更新して行動を決定。
        ※ このメソッドを呼ぶ前に、エピソード開始時に一度だけ
           self.q_network.reservoir.reset_internal_state() を呼ぶ必要がある。
        """
        # 1. オンライン用のforwardを呼び、リザバーの出力を得る
        #    この呼び出しにより、シミュレータの内部状態が更新される
        reservoir_output = self.q_network.reservoir.forward_online(state)
        
        # 2. 読み出し層でQ値を計算
        q_values = self.q_network.readout(reservoir_output)
        
        # ... (ε-greedy法で行動を決定) ...
        return action

    def update(self):
        """
        【オフライン処理】
        リプレイバッファからシーケンスを構築し、学習する。
        """
        # 1. リプレイバッファから遷移とインデックスをサンプリング
        transitions, indices = self.replay_buffer.sample(self.batch_size)
        
        # 2. シーケンスを構築 (前回の回答と同様)
        state_sequences = [self.replay_buffer.get_sequence(idx, self.sequence_length) for idx in indices]
        next_state_sequences = [self.replay_buffer.get_sequence(idx + 1, self.sequence_length) for idx in indices]

        # 3. 学習用のforward_sequenceメソッドを使ってQ値を計算
        # (ReservoirQNetworkクラスのget_q_valuesを修正して、内部でforward_sequenceを呼ぶようにする)
        q_values = self.q_network.get_q_values_for_training(state_sequences, 'main')
        # ... (以降のTD誤差計算、重み更新は同じ) ...