# 音声分類タスクを行うリザバーコンピューティング

PQNモデルによって実装したリザバーコンピューティングに音声分類タスクを行わせる

## 事前準備

audioファイルにti46/ti20にあるtrainとtestをフォルダごと配置する必要がある

## シミュレーション手順

下記はリザバー層の出力と線形学習器による計算を別々に行う方法
```bash
cd PQN_py/audio_rc
python make_cochleagram.py
cd PQN_py
python audio_rc/save_feature.py
python audio_rc/train_snn_readout.py
```


下記はリザバー層の出力を保存せず、一度に行う方法
```bash
cd PQN_py/audio_rc
python make_cochleagram.py
cd PQN_py
python audio_rc/train_snn_readout.py --mode snn
```
