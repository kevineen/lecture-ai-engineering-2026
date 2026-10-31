"""第2回向け: オリジナルデータセット作成・複数設定学習・結果比較 UI。

詳細: docs/usage/streamlit.md（個人用・gitignore。配置は docs/README.md）
CLI・依存: docs/usage/finetune.md

起動:
    uv sync --group ui --group finetune
    uv run streamlit run app/streamlit_app.py

学習タブは finetune group + CUDA torch が必要（片方だけ sync しない）。

構成:
    main() が各 render_*_tab() を呼ぶ。データ操作は app.finetune（ft）側の関数を
    呼ぶだけに寄せ、このファイルは UI 描画に専念する。一覧系の読み込みは
    @st.cache_data でキャッシュし、書き込み操作のあとに _refresh_caches() で
    失効させる。
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# streamlit run app/streamlit_app.py だと sys.path 先頭が app/ になり
# `import app` が失敗するため、リポジトリ根を先に入れる
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

import app.finetune as ft

# ホットリロード時に古い app.finetune が sys.modules に残ることがある
if not hasattr(ft, "get_vram_stats") or not hasattr(ft, "release_vram"):
    ft = importlib.reload(ft)

DEFAULT_WORK_DIR = Path("notebooks") / "qlora_lab"

DEFAULT_EDITOR_ROWS = [
    {
        "question": "課題の提出期限はいつですか",
        "answer": "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
        "needs_review": False,
    },
    {
        "question": "再提出はできますか",
        "answer": "この FAQ には記載がありません。運営に確認が必要です。",
        "needs_review": True,
    },
]

DEFAULT_HP_ROWS = [
    {"run_note": "baseline", "lora_r": 16, "lr": 2e-4, "num_train_epochs": 20, "seed": 42},
    {"run_note": "lora_r=8", "lora_r": 8, "lr": 2e-4, "num_train_epochs": 20, "seed": 42},
    {"run_note": "lr=1e-4", "lora_r": 16, "lr": 1e-4, "num_train_epochs": 20, "seed": 42},
    {"run_note": "epochs=10", "lora_r": 16, "lr": 2e-4, "num_train_epochs": 10, "seed": 42},
]


# ---------------------------------------------------------------------------
# キャッシュ付き読み込み（書き込み後に _refresh_caches() で失効させる）
# ---------------------------------------------------------------------------


@st.cache_data
def _cached_list_runs(work_dir: Path | None) -> list[dict[str, Any]]:
    return ft.list_runs(work_dir=work_dir)


@st.cache_data
def _cached_list_sft_datasets() -> list[dict[str, Any]]:
    return ft.list_sft_datasets()


@st.cache_data
def _cached_read_ui_train_history(limit: int = 30) -> list[dict[str, Any]]:
    return ft.read_ui_train_history(limit=limit)


@st.cache_data
def _cached_list_labs() -> list[dict[str, Any]]:
    return ft.list_labs()


def _refresh_caches() -> None:
    """書き込み操作のあとに一覧キャッシュを全部失効させる。"""
    _cached_list_runs.clear()
    _cached_list_sft_datasets.clear()
    _cached_read_ui_train_history.clear()
    _cached_list_labs.clear()


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------


def _safe_dataset_name(name: str) -> str:
    """ファイル名用スラッグ。"""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return cleaned.strip("_") or "custom_sft"


def _prefer_columns(df: pd.DataFrame, prefer: tuple[str, ...]) -> pd.DataFrame:
    """優先順に列を並べ直す（無い列は飛ばし、余った列は後ろに付ける）。"""
    cols = [c for c in prefer if c in df.columns] + [c for c in df.columns if c not in prefer]
    return df[cols]


def _row_to_config(row: dict[str, Any]) -> dict[str, Any]:
    """HP キュー行 / 過去 run の要約行 → train_lora 用 config（無い値は既定で埋める）。"""
    cfg = dict(ft.DEFAULT_CONFIG)
    cfg["lora_r"] = int(row.get("lora_r") or ft.DEFAULT_CONFIG["lora_r"])
    cfg["lr"] = float(row.get("lr") or ft.DEFAULT_CONFIG["lr"])
    cfg["num_train_epochs"] = int(
        row.get("num_train_epochs") or ft.DEFAULT_CONFIG["num_train_epochs"]
    )
    cfg["seed"] = int(row.get("seed") or ft.DEFAULT_CONFIG["seed"])
    return cfg


# ---------------------------------------------------------------------------
# サイドバー: VRAM（取得ボタン時のみ表示・自動更新しない）
# ---------------------------------------------------------------------------


def render_vram_sidebar() -> None:
    with st.sidebar:
        st.header("GPU / VRAM")
        st.caption("使用量は「取得」を押したときだけ更新します。")
        vb1, vb2 = st.columns(2)
        with vb1:
            fetch_vram = st.button("使用量を取得", key="vram_fetch", width="stretch")
        with vb2:
            free_vram = st.button("VRAM 解放", key="vram_free", width="stretch")

        if fetch_vram:
            st.session_state["vram_stats"] = ft.get_vram_stats()
            st.session_state["vram_fetched_at"] = datetime.now().isoformat(timespec="seconds")
        if free_vram:
            released = ft.release_vram()
            st.session_state["vram_stats"] = released.get("after") or {
                "ok": False,
                "error": released.get("error"),
            }
            st.session_state["vram_fetched_at"] = datetime.now().isoformat(timespec="seconds")
            if released.get("ok"):
                st.success("empty_cache / gc を実行しました")
            else:
                st.warning(released.get("error") or "解放に失敗しました")

        stats = st.session_state.get("vram_stats")
        if not stats:
            st.info("「使用量を取得」を押すと表示されます。")
        elif not stats.get("ok"):
            st.warning(stats.get("error") or "取得できませんでした")
        else:
            st.metric(
                "使用中 / 合計",
                f"{stats['used_gb']} / {stats['total_gb']} GB",
                f"{stats.get('used_pct')}%",
            )
            st.write(
                f"**{stats.get('name')}**  \n"
                f"空き {stats['free_gb']} GB  \n"
                f"torch allocated {stats['allocated_gb']} GB / "
                f"reserved {stats['reserved_gb']} GB"
            )
            if st.session_state.get("vram_fetched_at"):
                st.caption(f"取得時刻: {st.session_state['vram_fetched_at']}")
        st.caption(
            "解放はキャッシュ掃除です。学習中のモデル参照が残っていると"
            " allocated は大きく下がりません。"
        )


# ---------------------------------------------------------------------------
# ① オリジナルデータセット
# ---------------------------------------------------------------------------


def _render_dataset_memo_section(ds_rows: list[dict[str, Any]]) -> None:
    """既存データセット 1 件ごとのメモを選択・編集・保存する。"""
    if not ds_rows:
        return
    st.markdown("#### データセットごとのメモ")
    ds_options = [f"{r['name']} ({r['n']}件)" for r in ds_rows]
    ds_choice = st.selectbox("メモを書くデータセット", ds_options, key="ds_for_memo")
    name = ds_choice.split(" (", 1)[0]
    cur_memo = ft.load_dataset_memo(name)

    # データセット選択が変わったらメモ本文を読み直す
    if st.session_state.get("ds_memo_name") != name:
        st.session_state["ds_memo_name"] = name
        st.session_state["ds_memo_text"] = str(cur_memo.get("memo") or "")

    author = st.text_input("記録者名", value="", key="ds_memo_author", placeholder="例: tanaka")
    st.text_area(
        f"メモ: {name}",
        height=160,
        key="ds_memo_text",
        placeholder="生成元・件数の内訳・品質メモ・次に直したいことなど",
    )
    if st.button("データセットメモを保存", key="save_ds_memo_btn"):
        ft.save_dataset_memo(name, st.session_state["ds_memo_text"], author=author)
        _refresh_caches()
        st.success("データセットメモを保存しました。")
        st.rerun()
    st.caption(
        f"updated={cur_memo.get('updated_at') or '—'} / author={cur_memo.get('author') or '—'}"
    )


def _render_common_memo_section() -> None:
    """共通メモ（④ メモタブと同じ common_memo.json）を ① からも読み書きできるようにする。"""
    common_memo = ft.load_common_memo()
    # common_memo.json が変わっていたら（④ での保存・外部編集など）両タブの本文を読み直す。
    # updated_at は秒精度なので、同一秒内の保存にも追従できるよう中身ベースで判定する
    ver = json.dumps(common_memo, ensure_ascii=False, sort_keys=True)
    if st.session_state.get("common_memo_ver") != ver:
        st.session_state["common_memo_ver"] = ver
        body = str(common_memo.get("memo") or "")
        st.session_state["common_memo_text"] = body
        st.session_state["common_memo_text_ds"] = body

    st.markdown("#### 共通メモ（アイデア・雑記）")
    st.text_area(
        "共通メモ本文",
        height=160,
        key="common_memo_text_ds",
        placeholder="データセットのアイデア・やりたいこと・気づきなど。④ メモタブと共通",
    )
    if st.button("共通メモを保存", key="save_common_ds"):
        saved = ft.save_common_memo(
            st.session_state["common_memo_text_ds"],
            author=st.session_state.get("ds_memo_author", ""),
        )
        st.success(f"保存しました（更新: {saved.get('updated_at')}）。履歴は ④ メモタブで見られます。")
        st.rerun()
    st.caption(
        f"updated={common_memo.get('updated_at') or '—'} / "
        f"author={common_memo.get('author') or '—'} / 履歴は ④ メモタブ"
    )


def render_dataset_tab() -> None:
    st.subheader("独自 FAQ / Q&A から messages JSONL を作る")
    st.write(
        "質問・回答・`needs_review` を編集し、学習用の `messages` 形式 JSONL を保存します。"
        " FAQ 本文も差し替え可能です（宿題「実装」向け）。"
    )

    col_name, col_out = st.columns(2)
    with col_name:
        ds_name = st.text_input("データセット名", value="my_faq", key="ds_name")
    with col_out:
        out_path = st.text_input(
            "出力パス",
            value=str(ft.SFT_DIR / f"{_safe_dataset_name(ds_name)}.jsonl"),
            key="custom_out",
        )

    faq_body = st.text_area(
        "FAQ 本文（学習プロンプトに埋め込まれる）",
        value=ft.FAQ_TEXT,
        height=140,
        key="faq_body",
    )

    if "custom_editor_df" not in st.session_state:
        st.session_state.custom_editor_df = pd.DataFrame(DEFAULT_EDITOR_ROWS)

    edited = st.data_editor(
        st.session_state.custom_editor_df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "question": st.column_config.TextColumn("質問", required=True, width="large"),
            "answer": st.column_config.TextColumn("answer", required=True, width="large"),
            "needs_review": st.column_config.CheckboxColumn("needs_review", default=False),
        },
        key="custom_editor",
    )
    st.session_state.custom_editor_df = edited

    with st.expander("system プロンプト（参考・保存時は既定 SYSTEM を使用）"):
        st.code(ft.SYSTEM)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("JSONL を保存", type="primary", key="save_custom"):
            rows = edited.fillna("").to_dict(orient="records")
            path = ft.faq_editor_rows_to_jsonl(rows, Path(out_path), faq_text=faq_body)
            n = len(ft.read_jsonl(path))
            st.success(f"{n} 件 → {path.resolve()}")
            st.session_state["last_saved_dataset"] = str(path.resolve())
            _refresh_caches()
    with c2:
        if st.button("プレビュー（先頭2件）", key="preview_custom"):
            rows = edited.fillna("").to_dict(orient="records")
            previews = []
            for row in rows[:2]:
                q = str(row.get("question", "")).strip()
                if not q:
                    continue
                ex = ft.make_example(
                    q,
                    str(row.get("answer", "")).strip(),
                    bool(row.get("needs_review", False)),
                    faq_text=faq_body,
                )
                previews.append(ft.faq_example_to_messages(ex))
            st.json(previews)
    with c3:
        uploaded = st.file_uploader("既存 JSONL を取り込む", type=["jsonl", "json"], key="upload_jsonl")
        if uploaded is not None and st.button("アップロードを data/sft に保存", key="save_upload"):
            dest = ft.SFT_DIR / uploaded.name
            ft.SFT_DIR.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.getvalue())
            st.success(f"保存: {dest.resolve()}（{len(ft.read_jsonl(dest))} 件）")
            _refresh_caches()

    st.divider()
    st.subheader("既存データセット")
    ds_rows = _cached_list_sft_datasets()
    if ds_rows:
        st.dataframe(pd.DataFrame(ds_rows), width="stretch")
    else:
        st.info("`data/sft/` に JSONL がありません。")

    _render_dataset_memo_section(ds_rows)

    _render_common_memo_section()

    with st.expander("用語集 / 既定 FAQ から生成"):
        g_out = st.text_input(
            "glossary 出力", value=str(ft.SFT_DIR / "glossary_sft.jsonl"), key="g_out"
        )
        if st.button("glossary → JSONL", key="gen_glossary"):
            path = ft.make_dataset("glossary", Path(g_out))
            st.success(f"{len(ft.read_jsonl(path))} 件 → {path.resolve()}")
            _refresh_caches()
        if st.button("既定 FAQ 14 件 → JSONL", key="gen_faq"):
            path = ft.make_dataset("faq", ft.SFT_DIR / "faq_train.jsonl")
            st.success(f"{len(ft.read_jsonl(path))} 件 → {path.resolve()}")
            _refresh_caches()


# ---------------------------------------------------------------------------
# ② 学習（複数設定）
# ---------------------------------------------------------------------------


def _run_queue(
    queue: list[dict[str, Any]],
    *,
    work_dir: str,
    skip_eval: bool,
    train_jsonl: Path | None,
    lab_id: str | None,
    author: str,
    run_memo: str,
) -> None:
    """キューを上から順に学習し、全体進捗バーとログを表示する。"""
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ft.append_ui_train_history(
        {
            "batch_id": batch_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(train_jsonl) if train_jsonl else "builtin_faq",
            "work_dir": work_dir,
            "skip_eval": bool(skip_eval or train_jsonl),
            "lab_id": lab_id,
            "author": author,
            "queue": queue,
        }
    )
    _cached_read_ui_train_history.clear()

    overall = st.progress(0.0, text="準備中…")
    status = st.empty()
    log_box = st.empty()
    logs: list[str] = []
    results_ok: list[str] = []
    # 単一 run 内の進捗は全体バーの現在スロット内で上書きしない（slot で管理）
    slot = {"base": 0.0, "span": 1.0, "label": ""}

    def _on_progress(ev: dict[str, Any]) -> None:
        stage = ev.get("stage", "")
        msg = ev.get("message", "")
        frac = ev.get("progress")
        if msg:
            logs.append(f"{stage}: {msg}")
            log_box.code("\n".join(logs[-40:]))
        if frac is not None:
            overall.progress(
                min(1.0, slot["base"] + slot["span"] * float(frac)),
                text=f"{slot['label']} | {msg}",
            )

    for i, row in enumerate(queue):
        note = str(row.get("run_note", f"run{i}")).strip()
        slot["label"] = f"[{i + 1}/{len(queue)}] {note}"
        slot["base"] = i / len(queue)
        slot["span"] = 1 / len(queue)
        status.info(f"学習中: {note} ({i + 1}/{len(queue)})")
        overall.progress(i / len(queue), text=f"開始: {note}")

        cfg = _row_to_config(row)
        cfg["run_note"] = note

        try:
            result = ft.train_lora(
                cfg,
                work_dir=Path(work_dir),
                run_note=note,
                skip_eval=bool(skip_eval),
                train_jsonl=train_jsonl,
                progress_callback=_on_progress,
                lab_id=lab_id,
                memo=run_memo.strip() or None,
                memo_author=author.strip(),
            )
            rid = str(result.get("run_id"))
            results_ok.append(rid)
            logs.append(f"OK: {rid}")
            log_box.code("\n".join(logs[-40:]))
        except ImportError as e:
            st.error(
                "学習パッケージがありません。`uv sync --group finetune` と CUDA torch を入れてください。\n"
                f"{e}"
            )
            break
        except Exception as e:
            logs.append(f"ERROR ({note}): {e}")
            log_box.code("\n".join(logs[-40:]))
            st.error(f"{note} で失敗: {e}")
            with st.expander("詳細（traceback）"):
                st.code(traceback.format_exc())
            break
    else:
        overall.progress(1.0, text="キュー完了")
        status.success(f"完了 {len(results_ok)} 件: {', '.join(results_ok)}")
        st.session_state["last_compare_ids"] = results_ok
    _refresh_caches()


def render_train_tab() -> None:
    st.subheader("学習キュー（パラメータを複数行）")
    st.write(
        "各行が1実験です。上から順に学習し、進捗バーとログを更新します。"
        " 既定 FAQ を選ぶと学習前後の format/content 評価も入ります。"
        " オリジナル JSONL の場合は評価をスキップし、loss・学習時間を記録します。"
    )

    work_dir = st.text_input(
        "work_dir（チェックポイント等）",
        value=str(DEFAULT_WORK_DIR),
        key="train_work_dir",
    )
    work_path = Path(work_dir) if work_dir.strip() else None

    ds_options = ["[既定 FAQ・評価あり]"] + [
        f"{r['name']} ({r['n']}件)" for r in _cached_list_sft_datasets()
    ]
    ds_choice = st.selectbox("学習データ", ds_options, key="train_ds_choice")
    train_jsonl: Path | None = None
    if ds_choice != "[既定 FAQ・評価あり]":
        # name は "foo.jsonl (N件)" 形式
        name = ds_choice.split(" (", 1)[0]
        train_jsonl = ft.SFT_DIR / name

    skip_eval = st.checkbox(
        "評価をスキップ（高速・デバッグ用。カスタム JSONL では自動スキップ）",
        value=False,
        key="skip_eval",
    )

    labs = _cached_list_labs()
    lab_choices = ["（なし）"] + [str(x["lab_id"]) for x in labs]
    train_lab = st.selectbox(
        "紐づける実験 ID（lab）",
        lab_choices,
        key="train_lab_id",
        help="④ で作成した実験 ID。共同メモの単位になります。",
    )
    train_author = st.text_input("記録者名（メモ用）", value="", key="train_author")
    train_run_memo = st.text_area(
        "このキュー共通の run メモ（各実験に同じ文面を付ける）",
        value="",
        height=80,
        key="train_run_memo",
        placeholder="例: lr sweep / FAQ 既定データ / 期待: epochs で loss が下がる",
    )

    if "hp_queue_df" not in st.session_state:
        st.session_state.hp_queue_df = pd.DataFrame(DEFAULT_HP_ROWS)

    hp_edited = st.data_editor(
        st.session_state.hp_queue_df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "run_note": st.column_config.TextColumn("run_note", required=True),
            "lora_r": st.column_config.NumberColumn("lora_r", min_value=1, max_value=256, step=1),
            "lr": st.column_config.NumberColumn("lr", min_value=1e-6, max_value=1e-2, format="%.6f"),
            "num_train_epochs": st.column_config.NumberColumn("epochs", min_value=1, max_value=100, step=1),
            "seed": st.column_config.NumberColumn("seed", min_value=0, step=1),
        },
        key="hp_editor",
    )
    st.session_state.hp_queue_df = hp_edited

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("既定4行に戻す", key="reset_hp"):
            st.session_state.hp_queue_df = pd.DataFrame(DEFAULT_HP_ROWS)
            st.rerun()
    with b2:
        if st.button("過去 run から1行追加", key="add_from_past"):
            past = _cached_list_runs(work_path)
            if past:
                latest = past[-1]
                base = _row_to_config(latest)
                row = {
                    "run_note": f"from_{latest.get('run_note') or 'past'}",
                    "lora_r": base["lora_r"],
                    "lr": base["lr"],
                    "num_train_epochs": base["num_train_epochs"],
                    "seed": base["seed"],
                }
                st.session_state.hp_queue_df = pd.concat(
                    [st.session_state.hp_queue_df, pd.DataFrame([row])],
                    ignore_index=True,
                )
                st.rerun()
            else:
                st.warning("過去の run がありません。")

    st.divider()
    st.subheader("過去の学習設定")
    hist = _cached_read_ui_train_history(limit=30)
    past_runs = _cached_list_runs(work_path)
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        st.caption("UI から起動した学習キュー履歴")
        if hist:
            st.dataframe(pd.DataFrame(hist), width="stretch", height=220)
        else:
            st.info("まだ UI 学習履歴がありません。")
    with hcol2:
        st.caption("保存済み実験（outputs/finetune）")
        if past_runs:
            st.dataframe(
                _prefer_columns(
                    pd.DataFrame(past_runs),
                    (
                        "run_id",
                        "lab_id",
                        "run_note",
                        "memo_preview",
                        "lora_r",
                        "lr",
                        "num_train_epochs",
                        "dataset",
                        "last_loss",
                        "format_rate",
                        "content_rate",
                    ),
                ),
                width="stretch",
                height=220,
            )
        else:
            st.info("実験結果がありません。")

    st.divider()
    start = st.button("キューを上から学習開始", type="primary", key="start_train")
    if start:
        queue = hp_edited.dropna(how="all").to_dict(orient="records")
        queue = [r for r in queue if str(r.get("run_note", "")).strip()]
        if not queue:
            st.error("学習キューが空です。")
        else:
            lab_for_train = None if train_lab == "（なし）" else train_lab
            _run_queue(
                queue,
                work_dir=work_dir,
                skip_eval=bool(skip_eval),
                train_jsonl=train_jsonl,
                lab_id=lab_for_train,
                author=train_author,
                run_memo=train_run_memo,
            )


# ---------------------------------------------------------------------------
# ③ 結果比較
# ---------------------------------------------------------------------------


def render_compare_tab() -> None:
    st.subheader("実験結果の比較")
    work_hint = st.text_input(
        "追加で読む WORK ディレクトリ（任意）",
        value=str(DEFAULT_WORK_DIR),
        key="compare_work",
    )
    work_dir_cmp = Path(work_hint) if work_hint.strip() else None
    rows = _cached_list_runs(work_dir_cmp)

    if not rows:
        st.info(
            f"`{ft.FINETUNE_DIR}` に実験がありません。"
            "② の学習タブか `uv run python -m app.finetune train` で保存してください。"
        )
        return

    df = pd.DataFrame(rows)
    df = _prefer_columns(
        df,
        (
            "run_id",
            "lab_id",
            "run_note",
            "memo_preview",
            "dataset",
            "lora_r",
            "lr",
            "num_train_epochs",
            "format_rate",
            "content_rate",
            "last_loss",
            "training_seconds",
        ),
    )
    st.dataframe(df, width="stretch")

    run_ids = [str(r.get("run_id")) for r in rows if r.get("run_id")]
    default_pick = st.session_state.get("last_compare_ids") or run_ids[-min(4, len(run_ids)) :]
    default_pick = [r for r in default_pick if r in run_ids] or run_ids[:2]
    pick = st.multiselect("比較する run", run_ids, default=default_pick, key="compare_pick")

    if len(pick) >= 1:
        cmp_df = df[df["run_id"].isin(pick)].copy()
        st.write("選択した実験")
        st.dataframe(cmp_df, width="stretch")

        results = []
        for rid in pick:
            try:
                results.append(ft.load_run(rid))
            except FileNotFoundError:
                local = Path(work_hint) / "runs" / rid / "result.json"
                if local.is_file():
                    results.append(json.loads(local.read_text(encoding="utf-8")))
        if len(results) >= 2:
            diffs = ft.compare_config_diffs(results)
            st.write("config の差分キー")
            if diffs:
                st.json(diffs)
            else:
                st.caption("選択した実験の config に差分はありません。")

        results_by_id = {str(r.get("run_id")): r for r in results if r.get("run_id")}
        _render_output_comparison(pick, results_by_id)
        _render_cmp_lab_memo(pick, rows)

        detail_id = st.selectbox("result.json を表示", pick, key="detail_id")
        if detail_id:
            detail = next((r for r in results if r.get("run_id") == detail_id), None)
            if detail is None:
                try:
                    detail = ft.load_run(detail_id)
                except FileNotFoundError:
                    detail = None
            if detail:
                if detail.get("memo") or detail.get("lab_id"):
                    st.markdown(
                        f"**lab_id**: `{detail.get('lab_id') or '—'}`  \n"
                        f"**memo**: {detail.get('memo') or '（なし）'}"
                    )
                st.json(detail)
            else:
                st.warning("result.json を読めませんでした。")


def _row_for_question(rows: list[dict[str, Any]], question: str) -> dict[str, Any] | None:
    """評価行から質問テキストで該当行を探す。"""
    for row in rows:
        if str(row.get("question") or "") == question:
            return row
    return None


def _run_label(rid: str, results_by_id: dict[str, dict[str, Any]]) -> str:
    """比較表の列名・行名に使う run の表示名。"""
    return str((results_by_id.get(rid) or {}).get("run_note") or rid)


def _result_cell(row: dict[str, Any] | None) -> str:
    """評価行を ✓ / ✗ 表記にする。"""
    if row is None:
        return "—"
    f_ok, c_ok = bool(row.get("format_ok")), bool(row.get("content_ok"))
    if f_ok and c_ok:
        return "✓"
    parts = []
    if not f_ok:
        parts.append("形式✗")
    if not c_ok:
        parts.append("内容✗")
    return "+".join(parts)


def _color_status_cell(value: Any) -> str:
    """✓ / ✗ セルの着色（Styler 用）。"""
    text = str(value)
    if text == "✓":
        return "background-color: #e8f5e9; color: #1b5e20"
    if "✗" in text:
        return "background-color: #fbe9e7; color: #b71c1c"
    return ""


def _render_output_comparison(pick: list[str], results_by_id: dict[str, dict[str, Any]]) -> None:
    """選択 run の評価出力（生成テキスト）を質問を揃えた表で比較する。"""
    st.markdown("#### 出力結果の比較（質問揃え）")
    evals_by_id: dict[str, dict[str, Any]] = {}
    for rid in pick:
        evals = ft.load_run_eval(rid)
        if evals is not None:
            evals_by_id[rid] = evals
    if not evals_by_id:
        st.info(
            "選択した run に評価データ（eval_before / eval_after.json）がありません。"
            "既定 FAQ で学習した run を選ぶと出力を比較できます。"
        )
        return
    no_eval = [rid for rid in pick if rid not in evals_by_id]
    if no_eval:
        st.caption(f"評価データなし（比較対象外）: {', '.join(no_eval)}")
    active = [rid for rid in pick if rid in evals_by_id]

    # 質問の一覧（after の並び順を優先）
    questions: list[str] = []
    seen: set[str] = set()
    for rid in active:
        evals = evals_by_id[rid]
        for split in ("after", "before"):
            for row in (evals.get(split) or {}).get("rows") or []:
                q = str(row.get("question") or "")
                if q and q not in seen:
                    seen.add(q)
                    questions.append(q)
    if not questions:
        st.caption("評価行がありません。")
        return

    regressed_union: set[str] = set()
    for rid in active:
        result = results_by_id.get(rid) or {}
        regressed_union |= {str(q) for q in (result.get("regressed") or [])}

    # --- 全質問 × 選択 run の ✓/✗ 表（学習後・before は先頭 run のものを参考表示） ---
    st.caption("全質問の形式・内容判定。⚠ 付きの質問はいずれかの run で悪化しています。")
    matrix: list[dict[str, str]] = []
    for q in questions:
        row: dict[str, str] = {"質問": f"⚠ {q}" if q in regressed_union else q}
        before_cell = "—"
        for rid in active:
            b = _row_for_question(
                list((evals_by_id[rid].get("before") or {}).get("rows") or []), q
            )
            if b is not None:
                before_cell = _result_cell(b)
                break
        row["before"] = before_cell
        for rid in active:
            a = _row_for_question(
                list((evals_by_id[rid].get("after") or {}).get("rows") or []), q
            )
            row[_run_label(rid, results_by_id)] = _result_cell(a)
        matrix.append(row)
    df = pd.DataFrame(matrix)
    st.dataframe(
        df.style.map(_color_status_cell, subset=list(df.columns[1:])),
        width="stretch",
        height=min(720, 35 * (len(matrix) + 1) + 12),
    )

    # --- 選択した質問の返答比較 ---
    question = st.selectbox(
        "返答を見る質問",
        questions,
        format_func=lambda q: f"Q{questions.index(q) + 1}: {q}",
        key="cmp_question",
    )
    b_ref = None
    for rid in active:
        b_ref = _row_for_question(
            list((evals_by_id[rid].get("before") or {}).get("rows") or []), question
        )
        if b_ref is not None:
            break
    if b_ref is not None:
        st.caption("before（学習前・ベースモデル）の返答")
        st.code(str(b_ref.get("raw") or "（評価なし）"), language=None)

    def _md_cell(text: Any) -> str:
        return str(text or "").replace("|", "\\|").replace("\n", " ")

    lines = ["| run | 形式 | 内容 | after 返答 | 形式 NG 理由 |", "|---|---|---|---|---|"]
    for rid in active:
        result = results_by_id.get(rid) or {}
        a = _row_for_question(
            list((evals_by_id[rid].get("after") or {}).get("rows") or []), question
        )
        mark = " ⚠" if question in {str(q) for q in (result.get("regressed") or [])} else ""
        ok_f = "✓" if (a or {}).get("format_ok") else ("✗" if a else "—")
        ok_c = "✓" if (a or {}).get("content_ok") else ("✗" if a else "—")
        reason = str((a or {}).get("format_reason") or "—") if (a and not a.get("format_ok")) else "—"
        lines.append(
            f"| **{_run_label(rid, results_by_id)}**{mark} | {ok_f} | {ok_c} "
            f"| {_md_cell((a or {}).get('raw')) or '（評価なし）'} | {_md_cell(reason) or '—'} |"
        )
    st.markdown("\n".join(lines))


def _render_cmp_lab_memo(pick: list[str], rows: list[dict[str, Any]]) -> None:
    """選択 run が属する lab の共同メモを ③ から表示・編集できるようにする。"""
    st.markdown("#### 実験 ID（lab）の共同メモ")
    labs = _cached_list_labs()
    lab_ids = [str(x["lab_id"]) for x in labs]
    if not lab_ids:
        st.info(
            "実験 ID がまだありません。④ で作成するか、② 学習時に lab を選ぶと run が紐づきます。"
        )
        return
    picked_labs: list[str] = []
    for rid in pick:
        for r in rows:
            row_lab = str(r.get("lab_id") or "")
            if str(r.get("run_id")) == rid and row_lab and row_lab not in picked_labs:
                picked_labs.append(row_lab)
    sel_lab = st.selectbox(
        "実験 ID",
        lab_ids,
        index=lab_ids.index(picked_labs[0]) if picked_labs and picked_labs[0] in lab_ids else 0,
        key="cmp_lab_id",
        help="選択 run が属する lab を既定にします。④ と同じ lab.json を読み書きします。",
    )
    try:
        lab = ft.load_lab(sel_lab)
    except FileNotFoundError:
        st.warning("lab.json を読めませんでした。")
        return

    # lab の切替・外部（④ / CLI）での更新があったら本文を読み直す
    lab_ver = json.dumps(lab, ensure_ascii=False, sort_keys=True)
    if (
        st.session_state.get("cmp_lab_memo_lab") != sel_lab
        or st.session_state.get("cmp_lab_memo_ver") != lab_ver
    ):
        st.session_state["cmp_lab_memo_lab"] = sel_lab
        st.session_state["cmp_lab_memo_ver"] = lab_ver
        st.session_state["cmp_lab_memo_text"] = str(lab.get("shared_memo") or "")

    st.caption(
        f"title={lab.get('title') or '—'} / updated={lab.get('updated_at') or '—'} / "
        f"紐づく run: {len(lab.get('run_ids') or [])} 件"
    )
    st.code(str(lab.get("shared_memo") or "（まだ共同メモはありません）"), language=None)

    with st.expander("共同メモを編集・追記", expanded=False):
        st.text_area(
            "共同メモ本文",
            height=160,
            key="cmp_lab_memo_text",
            placeholder="方針・気づき・結論など。保存すると ④ の共同メモにも反映されます",
        )
        author = st.text_input("記録者名", value="", key="cmp_lab_memo_author", placeholder="例: tanaka")
        if st.button("共同メモを保存", key="save_cmp_lab_memo"):
            saved = ft.save_shared_memo(sel_lab, st.session_state["cmp_lab_memo_text"], author=author)
            _refresh_caches()
            st.success(f"保存しました（更新: {saved.get('updated_at')}）")
            st.rerun()

    log = list(lab.get("shared_log") or [])
    with st.expander(f"共同メモの履歴（{len(log)} 件）", expanded=False):
        if not log:
            st.caption("履歴はまだありません。")
        else:
            for entry in reversed(log[-20:]):
                st.markdown(f"**{entry.get('at')}** / {entry.get('author')} / `{entry.get('action')}`")
                st.code(str(entry.get("memo") or ""), language=None)


# ---------------------------------------------------------------------------
# ④ 実験 ID・メモ
# ---------------------------------------------------------------------------


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return str(value)


def _ok_badge(ok: Any) -> str:
    if ok is None:
        return "—"
    return "✓" if ok else "NG"


def _render_eval_viewer(run_id: str, *, memo_state_key: str) -> None:
    """run の評価出力（before / after の生成テキスト）を質問ごとに表示する。"""
    evals = ft.load_run_eval(run_id)
    if evals is None:
        st.info(
            "この run には評価データ（eval_before / eval_after.json）がありません。"
            "カスタム JSONL 学習・評価スキップの run です。"
        )
        return

    before = evals.get("before") or {}
    after = evals.get("after") or {}
    b_summary = before.get("summary") or {}
    a_summary = after.get("summary") or {}
    b_rows = list(before.get("rows") or [])
    a_rows = list(after.get("rows") or [])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("before 形式率", _fmt_rate(b_summary.get("format_rate")))
    m2.metric("before 内容率", _fmt_rate(b_summary.get("content_rate")))
    m3.metric("after 形式率", _fmt_rate(a_summary.get("format_rate")))
    m4.metric("after 内容率", _fmt_rate(a_summary.get("content_rate")))

    regressed: set[str] = set()
    try:
        detail = ft.load_run(run_id)
        regressed = {str(q) for q in (detail.get("regressed") or [])}
    except FileNotFoundError:
        pass
    if regressed:
        st.warning("悪化した質問（regressed）: " + " / ".join(sorted(regressed)))

    n = max(len(b_rows), len(a_rows))
    if n == 0:
        st.caption("評価行がありません。")
        return
    st.caption("質問ごとに展開して before / after の生成テキストを確認できます。")
    last_add = st.session_state.get("memo_last_add")
    for i in range(n):
        b = b_rows[i] if i < len(b_rows) else None
        a = a_rows[i] if i < len(a_rows) else None
        question = str((a or b or {}).get("question") or f"Q{i + 1}")
        mark = " ⚠" if question in regressed else ""
        with st.expander(f"Q{i + 1}{mark}: {question}", expanded=(i == 0 and last_add is None) or last_add == i):
            cb, ca = st.columns(2)
            with cb:
                st.markdown(
                    f"**before** 形式 {_ok_badge((b or {}).get('format_ok'))} / "
                    f"内容 {_ok_badge((b or {}).get('content_ok'))}"
                )
                st.code(str((b or {}).get("raw") or "（評価なし）"), language=None)
            with ca:
                st.markdown(
                    f"**after** 形式 {_ok_badge((a or {}).get('format_ok'))} / "
                    f"内容 {_ok_badge((a or {}).get('content_ok'))}"
                )
                st.code(str((a or {}).get("raw") or "（評価なし）"), language=None)
            notes: list[str] = []
            if a and not a.get("format_ok"):
                notes.append(f"after 形式 NG 理由: {a.get('format_reason') or '—'}")
            if (a or {}).get("gold_needs_review"):
                notes.append("gold: needs_review（根拠なし質問）")
            if notes:
                st.caption(" / ".join(notes))
            if st.button("メモに追記", key=f"memo_add_{run_id}_q{i}", help="メモ末尾にこの質問の行を追加"):
                st.session_state[memo_state_key] = (
                    str(st.session_state.get(memo_state_key) or "") + f"\n- Q{i + 1}（{question[:40]}）: "
                )
                st.session_state["memo_last_add"] = i
                st.rerun()


def _render_run_memo_editor(run_id: str, *, selected_lab: str, author: str) -> None:
    """左に評価出力ビューア、右に run メモ編集を並べて表示する。"""
    try:
        cur_memo = ft.load_run_memo(run_id)
    except FileNotFoundError:
        cur_memo = {"memo": "", "lab_id": selected_lab}

    # run 選択が変わったらメモ本文を読み直す（追記ボタンで書き換えるため state 管理する）
    if st.session_state.get("memo_run_id") != run_id:
        st.session_state["memo_run_id"] = run_id
        st.session_state["run_memo_text"] = str(cur_memo.get("memo") or "")
        st.session_state.pop("memo_last_add", None)

    left, right = st.columns([3, 2], gap="large")
    with left:
        _render_eval_viewer(run_id, memo_state_key="run_memo_text")
    with right:
        st.text_area(
            f"メモ: {run_id}",
            height=420,
            key="run_memo_text",
            placeholder="この run 固有の観察・失敗メモ・次に試すこと",
        )
        if st.button("各実験メモを保存", key="save_run_memo_btn"):
            ft.save_run_memo(
                run_id,
                st.session_state["run_memo_text"],
                lab_id=selected_lab,
                author=author,
            )
            _refresh_caches()
            st.success("run メモを保存しました。")
            st.rerun()
        st.caption(
            f"lab_id={cur_memo.get('lab_id') or '—'} / "
            f"updated={cur_memo.get('updated_at') or '—'} / "
            f"author={cur_memo.get('author') or '—'}"
        )


def render_memo_tab() -> None:
    st.subheader("実験 ID（lab）とメモ")
    st.write(
        "**共通メモ** は実験全体で共有する 1 枚メモ（履歴付き）、"
        " **実験 ID** は複数 run を束ねる単位です。"
        " **共同メモ** は lab 単位で共有・履歴付き、"
        " **各実験メモ** は `run_id` ごとに、評価出力を見ながら記録します。"
        f" 保存先: `{ft.FINETUNE_DIR / 'common_memo.json'}`、`{ft.FINETUNE_DIR / 'labs'}`、"
        f"各 `run` の `memo.json` / `result.json`。"
    )

    author = st.text_input("記録者名", value="", key="memo_author", placeholder="例: tanaka")

    st.markdown("#### 共通メモ（全実験共通）")
    common_memo = ft.load_common_memo()
    common_text = st.text_area(
        "共通メモ本文",
        height=180,
        key="common_memo_text",
        placeholder="データ方針・評価の約束・今週の focus など、実験全体で共有する内容",
    )
    if st.button("共通メモを保存", key="save_common"):
        saved = ft.save_common_memo(common_text, author=author)
        st.success(f"保存しました（更新: {saved.get('updated_at')}）")
        _refresh_caches()
        st.rerun()
    common_log = list(common_memo.get("log") or [])
    with st.expander(f"共通メモの履歴（{len(common_log)} 件）", expanded=False):
        if not common_log:
            st.caption("履歴はまだありません。")
        else:
            for entry in reversed(common_log[-20:]):
                st.markdown(
                    f"**{entry.get('at')}** / {entry.get('author')} / `{entry.get('action')}`"
                )
                st.code(str(entry.get("memo") or ""), language=None)

    st.markdown("#### 実験 ID の作成・選択")
    labs = _cached_list_labs()
    c_new1, c_new2, c_new3 = st.columns([2, 2, 1])
    with c_new1:
        new_lab_id = st.text_input("新しい実験 ID", value="lec02-hp-sweep", key="new_lab_id")
    with c_new2:
        new_lab_title = st.text_input("タイトル（任意）", value="", key="new_lab_title")
    with c_new3:
        st.write("")
        st.write("")
        if st.button("作成", type="primary", key="create_lab_btn"):
            lab = ft.create_lab(
                new_lab_id,
                title=new_lab_title or new_lab_id,
                author=author,
            )
            st.success(f"作成/取得: `{lab['lab_id']}`")
            st.session_state["active_lab_id"] = lab["lab_id"]
            _refresh_caches()
            st.rerun()

    if labs:
        st.dataframe(pd.DataFrame(labs), width="stretch")
    else:
        st.info("まだ実験 ID がありません。上で作成してください。")

    lab_ids = [str(x["lab_id"]) for x in labs]
    default_lab = st.session_state.get("active_lab_id")
    if default_lab not in lab_ids:
        default_lab = lab_ids[0] if lab_ids else None
    selected_lab = st.selectbox(
        "編集する実験 ID",
        lab_ids if lab_ids else ["（未作成）"],
        index=lab_ids.index(default_lab) if default_lab in lab_ids else 0,
        key="selected_lab",
    )

    if lab_ids and selected_lab in lab_ids:
        lab = ft.load_lab(selected_lab)
        st.session_state["active_lab_id"] = selected_lab

        # lab の切替・外部（③ の共同メモ保存 / CLI）での更新があったら本文を読み直す
        shared_ver = json.dumps(lab, ensure_ascii=False, sort_keys=True)
        if (
            st.session_state.get("shared_memo_lab") != selected_lab
            or st.session_state.get("shared_memo_ver") != shared_ver
        ):
            st.session_state["shared_memo_lab"] = selected_lab
            st.session_state["shared_memo_ver"] = shared_ver
            st.session_state["shared_memo_text"] = str(lab.get("shared_memo") or "")
            st.session_state["shared_title"] = str(lab.get("title") or selected_lab)

        st.markdown("#### 共同メモ（lab 共有）")
        shared_title = st.text_input("タイトル", key="shared_title")
        shared_text = st.text_area(
            "共同メモ本文",
            height=180,
            key="shared_memo_text",
            placeholder="方針・データ版・分担・気づきなど、チームで共有する内容",
        )
        if st.button("共同メモを保存", key="save_shared"):
            saved = ft.save_shared_memo(
                selected_lab,
                shared_text,
                author=author,
                title=shared_title,
            )
            st.success(f"保存しました（更新: {saved.get('updated_at')}）")
            _refresh_caches()
            st.rerun()

        log = list(lab.get("shared_log") or [])
        with st.expander(f"共同メモの履歴（{len(log)} 件）", expanded=False):
            if not log:
                st.caption("履歴はまだありません。")
            else:
                for entry in reversed(log[-20:]):
                    st.markdown(
                        f"**{entry.get('at')}** / {entry.get('author')} / `{entry.get('action')}`"
                    )
                    st.code(str(entry.get("memo") or ""), language=None)

        st.markdown("#### この実験 ID に紐づく run")
        linked = list(lab.get("run_ids") or [])
        all_runs = _cached_list_runs(None)
        all_ids = [str(r.get("run_id")) for r in all_runs if r.get("run_id")]
        if linked:
            linked_rows = [r for r in all_runs if r.get("run_id") in linked]
            if linked_rows:
                st.dataframe(
                    _prefer_columns(
                        pd.DataFrame(linked_rows),
                        (
                            "run_id",
                            "run_note",
                            "memo_preview",
                            "lora_r",
                            "lr",
                            "num_train_epochs",
                            "last_loss",
                            "format_rate",
                        ),
                    ),
                    width="stretch",
                )
            else:
                st.caption(", ".join(linked))
        else:
            st.caption("まだ紐づく run がありません。学習時に lab を選ぶか、下で紐づけてください。")

        link_run = st.selectbox(
            "既存 run をこの lab に紐づけ", ["（選択）"] + all_ids, key="link_run_sel"
        )
        if st.button("紐づけ実行", key="link_run_btn") and link_run != "（選択）":
            try:
                # save_run_memo が result 側 lab_id 更新 + lab への紐づけを兼ねる
                cur = ft.load_run_memo(link_run)
                ft.save_run_memo(
                    link_run,
                    str(cur.get("memo") or ""),
                    lab_id=selected_lab,
                    author=author or "link",
                )
            except FileNotFoundError:
                # result.json が無い run は lab 側だけ紐づける
                ft.link_run_to_lab(selected_lab, link_run)
            _refresh_caches()
            st.success(f"`{link_run}` → `{selected_lab}`")
            st.rerun()

        st.markdown("#### 各実験メモ（run 単位・出力を見ながら）")
        run_for_memo = st.selectbox(
            "メモを編集する run_id",
            linked if linked else all_ids,
            key="run_for_memo",
        )
        if run_for_memo:
            _render_run_memo_editor(run_for_memo, selected_lab=selected_lab, author=author)
    elif not lab_ids:
        st.warning("先に実験 ID を作成してください。")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="AIE2026 LoRA 実験", layout="wide")
    st.title("AIE2026 LoRA 実験管理")
    st.caption(
        "データセット / 複数学習 / 結果比較 / 実験ID・メモ。"
        " 学習には `uv sync --group ui --group finetune` と CUDA torch が必要です。"
    )
    render_vram_sidebar()
    tab_data, tab_train, tab_compare, tab_memo = st.tabs(
        ["① オリジナルデータセット", "② 学習（複数設定）", "③ 結果比較", "④ 実験ID・メモ"]
    )
    with tab_data:
        render_dataset_tab()
    with tab_train:
        render_train_tab()
    with tab_compare:
        render_compare_tab()
    with tab_memo:
        render_memo_tab()


main()
