class ReservoirQNetwork:
    # ... __init__ は同じ ...

    def get_q_values(self, state_sequence, network_type='main'):
        """
        引数が state から state_sequence に変わる
        """
        # 1. SNNリザバーを動かして、リザバーの状態を取得
        # バッチ処理の場合は、各シーケンスに対してループ処理が必要
        if isinstance(state_sequence, list) and isinstance(state_sequence[0], np.ndarray): # 単一シーケンス
             reservoir_state = self.reservoir.forward(state_sequence)
        else: # バッチ（シーケンスのリスト）
             reservoir_states_list = [self.reservoir.forward(seq) for seq in state_sequence]
             reservoir_state = np.array(reservoir_states_list)

        # 2. 読み出し層でQ値を計算 (この部分は変更なし)
        if network_type == 'main':
            q_values = self.readout(reservoir_state)
        else:
            with torch.no_grad():
                q_values = self.target_readout(reservoir_state)
        
        return q_values