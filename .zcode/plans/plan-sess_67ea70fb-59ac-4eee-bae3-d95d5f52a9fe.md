# Streamlit アプリのリファクタリング ＋ 「出力を見ながらメモ」機能

## Context

`app/streamlit_app.py`（733行）は実質トップレベル直列スクリプトで、関数は `_safe_dataset_name` の1つのみ。タブ②は約230行の `with` ブロックで、進捗クロージャが `st.session_state` 経由で外側と通信している。同じパターンの重複が複数箇所（「優先列で DataFrame 整形」×3、「行→config 抽出」×2、モジュール再束縛26行）、`@st.cache_data` 皆無で毎回ディスク再読みがある。

また各 run の生成出力の実体は `outputs/finetune/{run_id}/eval_before.json` / `eval_after.json`（`rows[].raw` に質問ごとの生テキスト、`format_ok` / `content_ok` 等付き）に保存されているのに、UI はどこでも読んでいないため、「結果の出力内容を見ながらメモを書く」が現在できない。

**目標**: ① 1ファイルのまま関数分割して保守性を上げる、② ④タブの run メモ編集を出力ビューアと横並びにしてメモ記入導線を一本化する。提出フロー（`WORK/result.json` スキーマ、`lectures/02/problem.md`）は変更しない。

## 変更ファイル

- `app/finetune.py` — データ層の小変更（UI 非依存・テスト可能）
- `app/streamlit_app.py` — リファクタリング本体＋新機能
- `tests/test_finetune.py` — 新関数のテスト
- `docs/usage/streamlit.md` / `docs/agent/project/finetune-context.md` — 追従更新（既存ファイル更新のみ、INDEX 新規登録は不要）

## 1. app/finetune.py（データ層）

1. **`load_run_eval(run_id, *, finetune_dir=FINETUNE_DIR) -> dict | None` を追加**
   - `outputs/finetune/{run_id}/eval_before.json` / `eval_after.json` を読み `{"before": {summary, rows} | None, "after": …}` を返す。両方無ければ `None`（skip_eval / カスタム JSONL run 向け）
   - `load_run`（L292）と同じ流儀で実装。`evaluate_model`（L828-902）が書く形式 `{"summary": …, "rows": [{question, raw, format_ok, format_reason, content_ok, both_ok, gold_needs_review}]}` が対象
2. **`list_sft_datasets`（L502）の件数カウントを全文 parse → 行カウントに**（`sum(1 for line in ... if line.strip())`。表示と件数は既存テスト `test_list_sft_datasets_and_ui_history` 互換のまま）
3. **`train_lora` に `memo_author: str = ""` kwarg を追加**（memo 保存 L1144-1161 に author を渡す）。UI 側の学習後の `save_run_memo` 再呼び出し（二重書き込み）をなくすため。後方互換（デフォルト値）

## 2. app/streamlit_app.py リファクタリング（1ファイル・関数分割）

- **モジュール再束縛26行（L37-62）を削除**し `import app.finetune as ft` のみに。各関数内で `ft.list_runs(...)` など属性アクセスで呼ぶ（ホットリロードガード L33-35 は現状維持で、再束縛が不要になる）
- **構成**（トップレベルは `main()` に集約し、ファイル末尾で呼ぶ。widget key は全て現状維持）:
  - ヘルパー: `_prefer_columns(df, prefer)`（3箇所の重複を解消）、`_row_to_config(row)`（L320-327 / L426-431 の重複を解消）
  - 定数: `DEFAULT_WORK_DIR = Path("notebooks") / "qlora_lab"`（L255 / L482 の重複解消）
  - `render_vram_sidebar()` / `render_dataset_tab()` / `render_train_tab()`（学習ループは `_run_queue()` ヘルパーへ。進捗の `run_frac_base` / `run_frac_span` / `run_label` の session_state 通信をクロージャのローカル変数に変更）/ `render_compare_tab()` / `render_memo_tab()`
- **キャッシュ**: `_cached_list_runs` / `_cached_list_sft_datasets` / `_cached_read_ui_train_history` / `_cached_list_labs` を `@st.cache_data` で定義し、書き込み操作後（データセット保存・学習完了・メモ/lab 保存・紐づけ）に `.clear()` を呼ぶ
- **細かい修正**: 学習失敗時の生 traceback（L467）を `st.expander` 内へ。④の紐づけ冗長（L684-696: `link_run_to_lab` → `save_run_memo`（内部で再 link）→ FileNotFoundError 時さらに link）を「`save_run_memo` を try、`FileNotFoundError` なら `link_run_to_lab` のみ」に簡素化

## 3. ④タブ: 出力を見ながらメモ編集（新機能）

`render_memo_tab()` 内の「各実験メモ（run 単位）」を2カラムに再構成:

- **左: 出力ビューア** — `ft.load_run_eval(run_id)` で読み、質問ごとに expander:
  - 質問文 ＋ before/after の `format_ok`/`content_ok` バッジ（✓/NG、NG は `format_reason` も表示）＋ `gold_needs_review`
  - before / after の `raw`（生生成テキスト）を2カラムの `st.code` で横並び表示
  - `result.json` の `regressed`（悪化質問）に該当する質問は expander タイトルに ⚠ を付けて上部に一覧表示
  - eval 無し（カスタム JSONL / skip_eval）の run は「評価データなし」の info
  - 各質問に「メモに追記」ボタン → `st.session_state["run_memo_text"] += f"\n- Q{n}: "` + `st.rerun()`
- **右: メモ編集** — 既存の `text_area` ＋ 保存ボタン ＋ メタ情報（lab_id/updated_at/author）caption はそのまま。text_area は `value` 渡しではなく session_state 管理に切り替え（run 選択変更時に初期化）してプログラム追記を可能にする

## 4. Docs 更新

- `docs/usage/streamlit.md`: ④タブに出力ビューア・メモ追記の説明を追記、関数分割構成に沿って記載微調整
- `docs/agent/project/finetune-context.md`: API 一覧に `load_run_eval`、`train_lora` の `memo_author`、Streamlit 構成（関数分割・キャッシュ）を反映
- 実装時に `docs/agent/INSTRUCTIONS.md` の配置ルールを確認（既存ファイル更新のみの予定）

## 5. テスト

- `tests/test_finetune.py` に追加:
  - `load_run_eval`: tmp な finetune_dir に eval_before/eval_after を書いて読み込み検証 ＋ 片方のみ ＋ 両方無し（None）の3ケース
  - 既存9テスト（`save_run_memo` / `train_lora` の memo 周り含む）が green であることを確認

## 制約事項（AGENTS.md 準拠）

- 提出物と個人 docs を混ぜない。`outputs/` は git 対象外のまま。コミットはユーザー指示時のみ
- Python 実行は `uv run ...`（`uv run streamlit run app/streamlit_app.py`）

## 検証

1. `uv run pytest -q`
2. `uv run python -m app.finetune list` / `uv run python -m app.finetune compare --work-dir notebooks\qlora_lab`（CLI が影響を受けていないこと）
3. `uv sync --group ui --group finetune` → `uv run streamlit run app/streamlit_app.py`
   - ④タブで評価ありの既存 run（例: `20261031_120905__lora_r_8`）を選び、出力ビューアの表示・「メモに追記」→ 保存 → 再読込を確認
   - ①〜③タブの表示がリファクタ前と同じ挙動であることを確認（学習実行は重いので省略可）