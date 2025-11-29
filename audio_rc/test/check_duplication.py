import os
import hashlib

def get_file_hash(filepath):
    """ファイルの中身を読み込んでハッシュ値(指紋)を返す関数"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # メモリ節約のため少しずつ読む
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

# ディレクトリ設定
train_dir = 'audio_rc/reservoir_inputs/train/coch_zero'
test_dir  = 'audio_rc/reservoir_inputs/test/coch_zero'

print("ファイルの中身(ハッシュ値)を計算中...")

# Trainデータのハッシュ集合を作る
train_hashes = set()
for fname in os.listdir(train_dir):
    full_path = os.path.join(train_dir, fname)
    if os.path.isfile(full_path):
        train_hashes.add(get_file_hash(full_path))

# Testデータのハッシュを計算しつつ、Trainにあるかチェック
overlap_count = 0
for fname in os.listdir(test_dir):
    full_path = os.path.join(test_dir, fname)
    if os.path.isfile(full_path):
        test_hash = get_file_hash(full_path)
        if test_hash in train_hashes:
            print(f"★重複発見！: {fname} (中身がTrainデータと同じです)")
            overlap_count += 1

if overlap_count == 0:
    print("【合格】中身が重複しているファイルはありませんでした。")
else:
    print(f"【警告】合計 {overlap_count} 個のファイルの中身が重複しています。")