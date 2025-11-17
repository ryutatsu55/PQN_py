# 音声分類タスクを行うリザバーコンピューティング

PQNモデルによって実装したリザバーコンピューティングに音声分類タスクを行わせる

## 実験手順

    cd PQN_py/audio_rc
    python make_cochleagram.py
    <!-- 上記操作はコクリアグラムを作成したいときのみ -->
    cd PQN_py
    python audio_rc/save_feature.py
    python audio_rc/train_snn_readout.py
