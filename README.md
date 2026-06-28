# xai-generics

このリポジトリは、MRI の画像変換と下流評価をローカルで再現するための作業用リポジトリです。  
現在は主に `T1 <-> T2` の 2D CycleGAN ベースラインを、ノートブック依存なしで回せる形に整理しています。

## まず見る場所

- `PROJECT_OVERVIEW.md`
- `docs/2D/spec.md`
- `docs/2D/plan.md`
- `docs/2D/tasks.md`
- `docs/2D/runbook.md`

## 主な構成

- `data_set/`: 元データ
- `ipynb/`: 参照用ノートブック
- `setting/`: ノートブックから抜き出した設定メモ
- `configs/2d/`: 実行用設定
- `scripts/`: 前処理、評価、後処理スクリプト
- `src/xai_generics/`: ローカル実装本体
- `outputs/`: 学習結果、生成結果、評価結果

## 実行の入口

環境変数 `UV_CACHE_DIR` をリポジトリ内に向けたうえで `uv run` を使います。

```powershell
$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'
uv run python -m xai_generics.cli.train_t1_t2 --dry-run
```

高速確認用の設定もあります。

```powershell
$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'
uv run python -m xai_generics.cli.train_t1_t2 --train-config configs/2d/train_t1_t2_cyclegan_fast.yaml --dataset-config configs/2d/dataset_t1_t2_fast.yaml --epochs 2 --max-samples 10 --validate-every-epoch
```

## 補足

- `training` は学習用データを指します。
- `test` は生成対象のデータを指します。ここでは `data_set/BraTS2017TestingData/` を使います。
- 画像評価は `RMSE`, `Mutual Information`, `PSNR`, `SSIM` を基本にしています。

