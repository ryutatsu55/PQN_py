class ReadoutLayer(nn.Module):
    """
    リザバーの状態（平均発火率ベクトル）を受け取り、Q値を計算する線形層。
    この層の重みだけが学習対象となる。
    """
    def __init__(self, reservoir_size, action_size):
        super().__init__()
        self.fc = nn.Linear(reservoir_size, action_size)

    def forward(self, reservoir_state):
        # PyCUDA/NumPyの出力をPyTorchテンソルに変換
        # バッチ処理のために、入力が単一のベクトルかバッチ（行列）かを確認
        if reservoir_state.ndim == 1:
            reservoir_state = torch.from_numpy(reservoir_state).float().unsqueeze(0) # バッチ次元を追加
        else:
            reservoir_state = torch.from_numpy(reservoir_state).float()
            
        # GPUが使えるなら .to(device) を追加
        # reservoir_state = reservoir_state.to(device)
        
        q_values = self.fc(reservoir_state)
        return q_values