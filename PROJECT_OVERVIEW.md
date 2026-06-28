# XAI Generics - プロジェクト概要

## 目的
このプロジェクトは、過去の MRI 合成と予後予測の研究を出発点として、以下を体系的に実験できる研究基盤へ作り直すことを目的とする。

- 異種モダリティ間の画像変換
- 予後予測を目的とした Radiomics
- 実画像と合成画像の両方を対象にした XAI
- RGB 的な 3 チャネル表現に縛られない多チャネル入力

目先の目標は、昔の CycleGAN 実験を再現すること自体ではない。  
モデル比較、アーキテクチャ改善、入力設計、XAI 解析を、再現可能な形で回せる研究環境を作ることが主目的である。

## ベースライン
このプロジェクトの初期ベースラインは、公開済み論文と当時の notebook 群で定義する。

### 論文ベースライン
- 論文: `13246_2024_Article_1443 (1).pdf`
- 題材: CycleGAN により生成した合成 MRI 画像から Radiomics 特徴を抽出し、膠芽腫の予後予測を行う
- 基本設定:
  - `T1w <-> T2w` の相互変換を CycleGAN で実施
  - 学習データは BraTS
  - 検証および Radiomics 解析は `TCGA-GBM / BraTS-TCGA-GBM`
  - Radiomics 特徴から予後予測モデルを構築
- 論文で明示されている主な limitation:
  - 主に `T1w` と `T2w` しか十分に使えていない
  - `CE-T1w` と `FLAIR` を十分に活用できていない
  - edema を含む腫瘍サブリージョン別の解析が不十分
  - 画像生成アーキテクチャが比較的単純
  - attention mechanism を導入できていない
  - clinical factors を将来的に統合すべき

### Notebook ベースライン
- `CycleGAN_batch1_RGB_T1_T2_.ipynb`
  - 2D の `T1 -> T2` 変換を行う基本的な CycleGAN 実験
- `CycleGAN_batch1_RGB_CET1_T1_to_T2.ipynb`
  - `CE-T1` と `T1` を pseudo-RGB 的に合成し、`T2` を生成する実験
- `CycleGAN_T12FLAIR.ipynb`
  - FLAIR を含む多モダリティ拡張の試行

これらの notebook は歴史的に重要であり、最初のベースラインとして扱う。  
ただし最終的なシステム形ではなく、再現可能なコードベースへ落とし直す前提の参照実装と位置づける。

## 中核となる研究課題

### RQ1. 合成 MRI は予後に有効な情報を本当に保持しているか
単に見た目が近いかではなく、予後予測に使える情報が保持されているかを検証する。

### RQ2. RGB への圧縮は不要なボトルネックではないか
RGB は人間の視覚都合の表現であり、機械学習モデルに 3 チャネル制約は必須ではない。  
`pseudo-RGB` よりも `N-channel 直接入力` のほうが有利ではないかを検証する。

### RQ3. 予後予測を実際に決めているのは画像のどこか、どの特徴か
知りたいのは次の 3 点である。

- どの領域が予測に効いているか
- どの Radiomics 特徴が効いているか
- 実画像と合成画像で explanation が一致するか

### RQ4. 文脈量やアーキテクチャの違いは重要か
以下を比較対象とする。

- 2D
- 2.5D
- 3D
- attention を含む派生モデル
- 多チャネル直接入力を扱う設計

## 主仮説
- 合成 MRI は、見た目の再現だけでなく、予後指向 Radiomics に必要な構造も保持しうる
- 単一モダリティ入力より multi-modal 入力のほうが下流の予後予測に有利である
- pseudo-RGB は便宜的表現に過ぎず、最適な学習表現とは限らない
- 複数モダリティや filtered image がある場合、`N-channel direct input + learned channel mixing` は固定 RGB 合成より有利である
- 実画像系と合成画像系で explanation が一致するかどうかは、単なる画像品質評価より強い検証になる

## 研究の主軸

### 1. 再現可能な学習基盤の整備
PyTorch もしくは PyTorch Lightning で、設定駆動の実験基盤を作る。

最低限必要な要素:
- 再現可能なデータ前処理
- 学習 / 評価 / 推論スクリプト
- checkpoint 管理
- 実験ログ記録
- バッチ実行
- 指標収集
- 旧 notebook ベースラインの再現

### 2. モデル比較
最低限、次を比較対象に入れる。

- CycleGAN
- pix2pix
- CUT
- VAE / latent translation 系ベースライン
- diffusion-based translation 系ベースライン

### 3. 入力表現の比較
次の入力設計を比較する。

- 1 チャネル入力
- pseudo-RGB 入力
- multi-modal direct tensor 入力
- 多チャネル filtered input
- `1x1 conv` や attention による学習型チャネル圧縮

### 4. 文脈量の比較
次を比較する。

- 2D slice-wise model
- 2.5D stacked-slice model
- 3D patch-based model

### 5. XAI / attribution 解析
古典的 XAI と deep XAI の両方を扱う。

Radiomics 側:
- feature importance
- SHAP
- coefficient stability
- modality ablation
- ROI ablation

Deep learning 側:
- Grad-CAM
- Integrated Gradients
- occlusion sensitivity
- attention map
- latent feature probing

最重要の問い:
- 実画像と合成画像が、予後に関して似た構造へ依存しているか

## 医療画像を超えた展開
RGB ボトルネックの問題は MRI 固有ではない。  
将来的には次の方向へ拡張できる。

- 非医療画像に対する多チャネル image translation
- hyperspectral 的入力
- radiomics 的な filtered tensor 入力
- 汎用 N-channel image translation framework

この広がりを意識して、プロジェクト名を `xai-generics` とする。

## 段階計画

### Phase 0. プロジェクト立ち上げ
- リポジトリ構成の作成
- config system の定義
- 環境整備とバッチ実行スクリプトの用意
- ベースライン資産の文書化

### Phase 1. ベースライン再現
- `T1 -> T2` CycleGAN の再現
- 可能な範囲で `CE-T1 + T1 -> T2` pseudo-RGB 実験の再現
- 過去実験の指標や出力と整合するか確認

### Phase 2. 実験基盤の改善
- modular dataloader
- Lightning module あるいは同等の trainer 設計
- 再利用可能な評価パイプライン
- 結果表と artifact 保存の仕組み

### Phase 3. 比較研究
- 複数の生成モデル比較
- `2D / 2.5D / 3D` 比較
- `1ch / RGB / Nch` 入力比較

### Phase 4. XAI 研究
- 予後予測モデルの explainability を追加
- Radiomics importance と image-space saliency の対応付け
- 実画像と合成画像で explanation の一致度を比較

### Phase 5. 多モダリティ・多チャネル拡張
- `CE-T1w`, `FLAIR`, edema-aware region
- filtered volume
- 大規模チャネル入力実験

## 最初の成果物
- 再現可能なベースラインコードベース
- 1 本のきれいな CycleGAN 再現ベンチマーク
- モデル比較の構造化された実験表
- 領域レベルと特徴レベルの重要度を示す XAI レポート雛形

## 想定ディレクトリ構成
初期案であり、今後変更してよい。

```text
xai-generics/
  docs/
    PROJECT_OVERVIEW.md
    RESEARCH_QUESTIONS.md
    PAPER_NOTES.md
    BASELINE_NOTEBOOKS.md
  configs/
  data/
  scripts/
    prepare/
    train/
    eval/
  src/
    datamodules/
    models/
    losses/
    metrics/
    xai/
    radiomics/
    experiments/
  notebooks/
    exploratory/
  outputs/
```

## 直近の次ステップ
論文と notebook を、正式な実験仕様へ落とし込む。

- 必要なデータセットは何か
- どの前処理をしていたか
- ベースラインでどのコマンドを実行していたか
- 最初に再現すべき結果は何か

ここまで固定できれば、当時の研究意図を壊さずに実装へ進める。
