# RNN Reservoir Computing Simulation Framework

Spiking Neural Network (SNN) およびリザバーコンピューティングを用いた、空間・時空間パターン認識タスクのシミュレーションフレームワークです。

## 🚀 特徴

* **設定の一元管理**: `src/RNN_config.py` 内の `Config` クラスですべての実験パラメータ（ニューロン数、入力強度、パス等）を管理します。
* **実験の自動化**: シェルスクリプト `run_experiment.sh` により、データ生成から学習、評価までを一括実行します。
* **再現性の確保**: 実験実行ごとに結果フォルダを自動生成し、その時点の `RNN_config.py` をスナップショットとして自動保存します。

## 📂 ディレクトリ構成

```text
.
├── run_experiment.sh           # 自動化・管理用スクリプト（メインの実行ファイル）
├── make_spatial_input.py       # 入力データ生成スクリプト
├── src/
│   ├── RNN_config.py           # 【重要】設定一元管理ファイル
│   ├── PQN_RNN_onGPU.py        # SNNコアロジック (GPU)
│   └── ...
└── RNN_analyze/
    ├── spatial_recognition.py  # 空間認識タスク (SNN -> Feature)
    ├── spatiotemp_recognition.py # 時空間認識タスク (Feature -> Readout)
    ├── reservoir_inputs/       # 生成された入力データ (自動生成)
    ├── reservoir_outputs/      # 中間出力データ
    └── results/                # 【重要】実験結果の保存先
```

**自動実行**
```bash
# 実行権限を付与（初回のみ）
chmod +x run_experiment.sh

# 実験を開始
./run_experiment.sh
```

```bash
# メモをつける
./run_experiment.sh memomemohogehoge
```

**手動実行**
```bash
# 1. データ生成
python make_spatial_input.py

# 2. 空間認識タスク (SNN実行)
python RNN_analyze/spatial_recognition.py --mode snn

# 3. 時空間認識タスク (読み出し層学習)
python RNN_analyze/spatiotemp_recognition.py --mode feature
```
