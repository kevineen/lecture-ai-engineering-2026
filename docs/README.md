# ドキュメント配置（ローカル用）

このリポジトリの **講座提出物（`lectures/`・`app/` のコードなど）とは別に**、手元の使い方メモやエージェント向けコンテキストを置く場所です。

| パス | 用途 | git |
|---|---|---|
| [`docs/README.md`](README.md)（本ファイル） | 配置ルールの説明 | **追跡する** |
| [`AGENTS.md`](../AGENTS.md)（リポジトリ直下） | エージェント向け入口（Cursor 等が参照） | **無視** |
| [`docs/usage/`](usage/) | 人間向けの使い方・手順・メモ | **無視** |
| [`docs/agent/INSTRUCTIONS.md`](agent/INSTRUCTIONS.md) | **docs 構造更新時にエージェントが読む指示** | **無視** |
| [`docs/agent/project/`](agent/project/) | このリポ実装のエージェント用コンテキスト | **無視** |
| [`docs/agent/study/`](agent/study/) | Hermes / Pi などエージェント学習資料 | **無視** |

## なぜ `AGENTS.md` はトップか

Cursor など多くのツールは **リポジトリ直下の `AGENTS.md`** をプロジェクト指示として読みます。  
docs の詳細ルールは `docs/agent/INSTRUCTIONS.md`、学習メモは `docs/agent/study/` に分けます。

## Git ブランチ方針（要約）

| ブランチ | 用途 |
|---|---|
| `main` / `lecNN` | 講座提出。提出後はその回の提出物を触らない |
| `personal/lab` | 個人拡張を使い回し・更新。提出 PR には載せない |

手順の詳細は [`usage/git-workflow.md`](usage/git-workflow.md)（gitignore・ローカル用）。

## 更新・追加のルール

エージェントが構造を変えるときは必ず [`docs/agent/INSTRUCTIONS.md`](agent/INSTRUCTIONS.md) に従う。要約:

1. **使い方・手順** → `docs/usage/` + `INDEX.md` 更新
2. **短い方針** → ルート `AGENTS.md`
3. **このリポの実装文脈** → `docs/agent/project/` + `project/INDEX.md`
4. **Hermes / Pi 等の勉強** → `docs/agent/study/` + `study/INDEX.md`
5. **講座の正式課題文** → `lectures/NN/problem.md`（**編集禁止**。提出で書くのは `answer.md`）
6. 個人用詳細は提出 PR に含めない（gitignore）
7. 個人拡張のコード変更は `personal/lab`。提出ブランチ `lecNN` に混ぜない

## 初期ファイル（ローカル・git 対象外の例）

- `AGENTS.md` … エージェント入口（第2回: finetune / Streamlit / uv group 注意を含む）
- `docs/usage/git-workflow.md` … 提出ブランチと `personal/lab` の運用
- `docs/usage/lec02-homework.md` … 第2回：演習〜宿題（基礎/実装/発展）。各段階で提出可
- `docs/usage/finetune.md` … CLI・依存・HP 比較・result.json
- `docs/usage/streamlit.md` … UI・トラブルシュート
- `docs/usage/INDEX.md`
- `docs/agent/INSTRUCTIONS.md` / `README.md`
- `docs/agent/project/finetune-context.md` … 実装 API・設計方針
- `docs/agent/study/hermes.md` / `pi.md` / `comparison.md`
