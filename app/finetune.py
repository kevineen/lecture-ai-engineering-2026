"""第2回: LoRA ファインチューニングと実験管理。

詳細な使い方（人間向け）: docs/usage/finetune.md / docs/usage/streamlit.md
エージェント向け: AGENTS.md（ルート） / docs/agent/INSTRUCTIONS.md / docs/agent/project/
（いずれも個人用・gitignore。配置ルールは docs/README.md）

使い方（要約）:
    uv run python -m app.finetune make-dataset --source glossary -o data/sft/glossary_sft.jsonl
    uv run python -m app.finetune list
    uv run python -m app.finetune compare
    uv run python -m app.finetune train --note baseline
    uv run python -m app.finetune train --note custom --train-jsonl data/sft/my_faq.jsonl
    uv sync --group ui --group finetune
    uv run streamlit run app/streamlit_app.py

ノートブックからも build_result / save_run / extract_last_loss を import できる。
Colab ではリポジトリが無いことがあるので、ノートブック側にも同内容の保存セルがある。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

# リポジトリ直下
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
FINETUNE_DIR = OUTPUTS_DIR / "finetune"
LABS_DIR = FINETUNE_DIR / "labs"
SFT_DIR = REPO_ROOT / "data" / "sft"
GLOSSARY_PATH = REPO_ROOT / "data" / "glossary.json"

# ノートブックと同じ既定 CONFIG（パス系は呼び出し側で上書き）
DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},
    "run_note": "baseline",
}

# FAQ 実験用の定数（notebooks/lora_format_tuning.ipynb と揃える）
FAQ = {
    "deadline": "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
    "destination": "提出先は学習ポータルの「課題」欄です。",
    "format": "提出形式は PDF 1 ファイルで、ファイル名は 学籍番号_課題番号.pdf です。",
    "office_hours": "質問は毎週火曜 17:00〜18:00 のオフィスアワーで受け付けます。",
    "late": "期限後の提出は、期限から 48 時間以内なら減点のうえ受け付けます。",
}
FAQ_TEXT = "\n".join(f"- {v}" for v in FAQ.values())
GROUNDED = {
    "deadline": (
        ["課題の提出期限はいつですか", "いつまでに出せばいいですか", "締切を教えてください", "提出の期限は？"],
        ["9 月 30 日", "23:59"],
    ),
    "destination": (
        ["提出先はどこですか", "どこにアップロードしますか", "課題はどこから出しますか", "提出場所は？"],
        ["学習ポータル", "課題"],
    ),
    "format": (
        ["提出形式を教えてください", "ファイル名の決まりはありますか", "何のファイル形式で出しますか", "PDF 以外でも出せますか"],
        ["PDF"],
    ),
    "office_hours": (
        ["質問はいつできますか", "オフィスアワーはいつですか", "先生に質問する時間はありますか", "相談できる時間帯は？"],
        ["火曜", "17:00"],
    ),
    "late": (
        ["遅れて出したらどうなりますか", "期限を過ぎた場合は受け付けますか", "遅延提出のルールは？", "締切後の扱いを教えてください"],
        ["48 時間", "減点"],
    ),
}
UNGROUNDED = [
    "再提出はできますか",
    "欠席した回の扱いを教えてください",
    "他の受講者の点数は見られますか",
    "成績はいつ出ますか",
    "提出期限は延長できますか",
    "グループでの提出は可能ですか",
    "課題の配点はどうなっていますか",
    "講義の録画は公開されますか",
]
TEST_EXTRA_GROUNDED = {
    "deadline": ["提出期限の日付と時刻を教えてください", "課題の締切は何月何日の何時ですか"],
    "destination": ["課題の提出はどこで行いますか", "提出物を出す先を教えてください"],
    "format": ["提出するファイルの形式は何ですか", "提出ファイルは何形式で、いくつ出しますか"],
    "office_hours": ["オフィスアワーの曜日と時間を教えてください", "質問を受け付けてもらえるのは何曜日の何時からですか"],
    "late": ["提出が締切に間に合わなかったらどうなりますか", "期限後に提出した課題は受理されますか"],
}
TEST_EXTRA_UNGROUNDED = [
    "課題の文字数に制限はありますか",
    "提出した課題へのフィードバックはもらえますか",
    "参考文献の書き方に指定はありますか",
    "オフィスアワーは事前予約が必要ですか",
]
UNGROUNDED_ANSWER = "この FAQ には記載がありません。運営に確認が必要です。"

SYSTEM = (
    "あなたは講座 FAQ の回答係です。与えられた FAQ だけを根拠に、必ず JSON 1 行で答えてください。"
    '形式: {"answer": 文字列, "needs_review": 真偽値}。'
    "FAQ に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"
)

GLOSSARY_TEMPLATES = ["「{term}」とは何ですか？", "{term}について簡単に説明してください。"]

# result.json の config から除外するパス系キー
_CONFIG_SKIP_KEYS = frozenset({"work_dir", "drive_dir", "memo"})


def _safe_slug(text: str) -> str:
    """ファイル名用に安全な文字列にする。"""
    slug = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    return slug.strip("_")[:80] or "run"


def make_run_id(note: str = "baseline", when: datetime | None = None) -> str:
    """タイムスタンプ + note から run_id を作る。"""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}__{_safe_slug(note)}"


def extract_last_loss(log_history: list[dict[str, Any]] | None) -> float | None:
    """Trainer の log_history から最後の loss を取り出す。無ければ None。"""
    if not log_history:
        return None
    for entry in reversed(log_history):
        if "loss" in entry:
            try:
                return float(entry["loss"])
            except (TypeError, ValueError):
                return None
    return None


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """提出・比較用にパス系キーを除いた config を返す。"""
    return {k: v for k, v in config.items() if k not in _CONFIG_SKIP_KEYS}


def build_result(
    *,
    config: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    train_info: dict[str, Any] | None = None,
    regressed: list[str] | None = None,
    knowledge: list[dict[str, Any]] | None = None,
    versions: dict[str, str] | None = None,
    method: str = "LoRA（量子化なし）",
    executed_training_myself: bool = True,
    run_id: str | None = None,
    run_note: str | None = None,
    lab_id: str | None = None,
    memo: str | None = None,
) -> dict[str, Any]:
    """拡張 result.json スキーマの dict を組み立てる。"""
    note = run_note if run_note is not None else str(config.get("run_note", "baseline"))
    rid = run_id or make_run_id(note)
    result: dict[str, Any] = {
        "run_id": rid,
        "run_note": note,
        "lab_id": lab_id or config.get("lab_id"),
        "memo": memo if memo is not None else str(config.get("memo") or ""),
        "config": public_config(config),
        "versions": versions or {},
        "method": method,
        "train": train_info
        or {
            "last_loss": None,
            "training_seconds": None,
            "global_step": None,
            "epoch": None,
        },
        "before": before,
        "after": after,
        "regressed": list(regressed or []),
        "knowledge": list(knowledge or []),
        "executed_training_myself": executed_training_myself,
    }
    return result


def run_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    """index.jsonl / 比較表用の1行要約。"""
    cfg = result.get("config") or {}
    train = result.get("train") or {}
    before = result.get("before") or {}
    after = result.get("after") or {}
    memo = str(result.get("memo") or "")
    return {
        "run_id": result.get("run_id"),
        "run_note": result.get("run_note"),
        "lab_id": result.get("lab_id") or cfg.get("lab_id"),
        "memo": memo,
        "memo_preview": (memo[:80] + "…") if len(memo) > 80 else memo,
        "lora_r": cfg.get("lora_r"),
        "lr": cfg.get("lr"),
        "num_train_epochs": cfg.get("num_train_epochs"),
        "seed": cfg.get("seed"),
        "dataset": cfg.get("dataset"),
        "format_rate_before": before.get("format_rate"),
        "content_rate_before": before.get("content_rate"),
        "format_rate": after.get("format_rate"),
        "content_rate": after.get("content_rate"),
        "both_rate": after.get("both_rate"),
        "last_loss": train.get("last_loss"),
        "training_seconds": train.get("training_seconds"),
        "executed_training_myself": result.get("executed_training_myself"),
    }


def save_run(
    result: dict[str, Any],
    *,
    work_dir: Path | None = None,
    finetune_dir: Path = FINETUNE_DIR,
    copy_eval_from: Path | None = None,
) -> Path:
    """result.json を提出用 work_dir と outputs/finetune/{run_id}/ に保存する。

    Returns:
        outputs/finetune/{run_id}/ のパス（実験管理の正本）。
    """
    run_id = str(result.get("run_id") or make_run_id(str(result.get("run_note", "run"))))
    result = {**result, "run_id": run_id}

    run_dir = finetune_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (run_dir / "result.json").write_text(result_text, encoding="utf-8")

    # 提出用の最新1件
    if work_dir is not None:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        (work / "result.json").write_text(result_text, encoding="utf-8")
        # 同セッション内の複数実験比較用
        runs_local = work / "runs" / run_id
        runs_local.mkdir(parents=True, exist_ok=True)
        (runs_local / "result.json").write_text(result_text, encoding="utf-8")

    # eval_*.json があれば実験フォルダへコピー
    src = Path(copy_eval_from) if copy_eval_from is not None else work_dir
    if src is not None:
        for name in ("eval_before.json", "eval_after.json"):
            p = Path(src) / name
            if p.is_file():
                shutil.copy2(p, run_dir / name)

    # index.jsonl に追記（同一 run_id は置き換え）
    _append_index(run_summary_row(result), finetune_dir=finetune_dir)
    return run_dir


def _append_index(row: dict[str, Any], *, finetune_dir: Path = FINETUNE_DIR) -> None:
    """index.jsonl を更新（同じ run_id があれば上書き）。"""
    finetune_dir.mkdir(parents=True, exist_ok=True)
    index_path = finetune_dir / "index.jsonl"
    rows: list[dict[str, Any]] = []
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                prev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if prev.get("run_id") != row.get("run_id"):
                rows.append(prev)
    rows.append(row)
    index_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def load_run(run_id: str, *, finetune_dir: Path = FINETUNE_DIR) -> dict[str, Any]:
    """outputs/finetune/{run_id}/result.json を読む。"""
    path = finetune_dir / run_id / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"result.json がありません: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_eval(
    run_id: str,
    *,
    finetune_dir: Path = FINETUNE_DIR,
) -> dict[str, dict[str, Any] | None] | None:
    """run の評価詳細（eval_before.json / eval_after.json）を読む。

    evaluate_model が書いた {"summary": …, "rows": […]} をそのまま返す。
    両方無ければ None（評価スキップ・カスタム JSONL 学習の run など）。
    """
    finetune_dir = Path(finetune_dir)

    def _read(label: str) -> dict[str, Any] | None:
        path = finetune_dir / run_id / f"eval_{label}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    before = _read("before")
    after = _read("after")
    if before is None and after is None:
        return None
    return {"before": before, "after": after}


def list_runs(*, finetune_dir: Path = FINETUNE_DIR, work_dir: Path | None = None) -> list[dict[str, Any]]:
    """実験要約の一覧。index.jsonl を優先し、無ければ各 run の result.json を走査。"""
    rows: list[dict[str, Any]] = []
    index_path = finetune_dir / "index.jsonl"
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    elif finetune_dir.is_dir():
        for result_path in sorted(finetune_dir.glob("*/result.json")):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                rows.append(run_summary_row(result))
            except (json.JSONDecodeError, OSError):
                continue

    # Colab / ローカル notebook の WORK/runs も拾う
    if work_dir is not None:
        for result_path in sorted(Path(work_dir).glob("runs/*/result.json")):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                row = run_summary_row(result)
                if not any(r.get("run_id") == row.get("run_id") for r in rows):
                    rows.append(row)
            except (json.JSONDecodeError, OSError):
                continue
    return rows


def compare_summaries(
    runs: list[dict[str, Any]] | None = None,
    *,
    finetune_dir: Path = FINETUNE_DIR,
    work_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """比較表用の行リスト（未指定なら list_runs）。"""
    return runs if runs is not None else list_runs(finetune_dir=finetune_dir, work_dir=work_dir)


def _config_value_key(value: Any) -> str:
    """dict/list を含む config 値を、差分比較用のハッシュ可能なキーにする。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def compare_config_diffs(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """複数 result の config で値が分かれるキーだけを返す。"""
    if not results:
        return {}
    keys: set[str] = set()
    for r in results:
        keys.update((r.get("config") or {}).keys())
    diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        # gen など dict 値もあるので、JSON 化して一意性を見る
        unique_vals = {
            _config_value_key((r.get("config") or {}).get(key)) for r in results
        }
        if len(unique_vals) > 1:
            diffs[key] = {str(r.get("run_id")): (r.get("config") or {}).get(key) for r in results}
    return diffs


# ---------------------------------------------------------------------------
# データセット
# ---------------------------------------------------------------------------


def make_example(question: str, answer: str, needs_review: bool, faq_text: str = FAQ_TEXT) -> dict[str, Any]:
    """ノートブックと同じ FAQ 学習例。"""
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {
        "question": question,
        "faq": faq_text,
        "target": target,
        "needs_review": needs_review,
    }


def build_faq_splits() -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    """ノートブックと同じ train / valid / test / gold を返す。"""

    def split_templates(templates: list[str]) -> tuple[list[str], list[str], list[str]]:
        return templates[:2], templates[2:3], templates[3:4]

    train: list[dict] = []
    valid: list[dict] = []
    test: list[dict] = []
    gold: dict[str, dict] = {}
    for key, (templates, must_include) in GROUNDED.items():
        tr, va, te = split_templates(templates)
        for q in tr:
            train.append(make_example(q, FAQ[key], False))
        for q in va:
            valid.append(make_example(q, FAQ[key], False))
        for q in te + TEST_EXTRA_GROUNDED[key]:
            test.append(make_example(q, FAQ[key], False))
            gold[q] = {"needs_review": False, "must_include": must_include}
    for i, q in enumerate(UNGROUNDED):
        ex = make_example(q, UNGROUNDED_ANSWER, True)
        bucket = [train, train, valid, test][i % 4]
        bucket.append(ex)
        if bucket is test:
            gold[q] = {"needs_review": True, "must_include": ["確認"]}
    for q in TEST_EXTRA_UNGROUNDED:
        test.append(make_example(q, UNGROUNDED_ANSWER, True))
        gold[q] = {"needs_review": True, "must_include": ["確認"]}
    return train, valid, test, gold


def iter_glossary_sft_records(
    glossary_path: Path = GLOSSARY_PATH,
    templates: list[str] | None = None,
) -> list[dict[str, Any]]:
    """用語集から messages 形式の SFT レコード一覧を作る。"""
    data = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
    terms = data["terms"]
    tpls = templates or GLOSSARY_TEMPLATES
    records: list[dict[str, Any]] = []
    for t in terms:
        for template in tpls:
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": template.format(term=t["term"])},
                        {"role": "assistant", "content": t["definition"]},
                    ]
                }
            )
    return records


def faq_example_to_messages(example: dict[str, Any]) -> dict[str, Any]:
    """FAQ 例を chat messages（学習用）に変換。"""
    user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": example["target"]},
        ]
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    """JSONL を書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL を読む。空行は無視。"""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def make_dataset(
    source: str,
    output: Path,
    *,
    glossary_path: Path = GLOSSARY_PATH,
) -> Path:
    """glossary / faq から SFT JSONL を生成して保存する。"""
    source = source.lower().strip()
    if source == "glossary":
        records = iter_glossary_sft_records(glossary_path)
    elif source == "faq":
        train, _, _, _ = build_faq_splits()
        records = [faq_example_to_messages(ex) for ex in train]
    else:
        raise ValueError(f"未知の source: {source}（glossary / faq）")
    return write_jsonl(records, output)


def faq_editor_rows_to_jsonl(rows: list[dict[str, Any]], path: Path, faq_text: str = FAQ_TEXT) -> Path:
    """Streamlit エディタ行（question, answer, needs_review）を messages JSONL にする。"""
    records: list[dict[str, Any]] = []
    for row in rows:
        q = str(row.get("question", "")).strip()
        if not q:
            continue
        answer = str(row.get("answer", "")).strip()
        needs = bool(row.get("needs_review", False))
        ex = make_example(q, answer, needs, faq_text=faq_text)
        records.append(faq_example_to_messages(ex))
    return write_jsonl(records, path)


def _dataset_memo_path(name: str, *, sft_dir: Path = SFT_DIR) -> Path:
    """データセット（data/sft の JSONL）に対応するメモ sidecar のパス。"""
    return Path(sft_dir) / "memos" / f"{Path(name).stem}.memo.json"


def save_dataset_memo(
    name: str,
    memo: str,
    *,
    author: str = "",
    sft_dir: Path = SFT_DIR,
) -> dict[str, Any]:
    """データセットごとのメモを上書き保存する（run の memo.json と同じ単純形式）。"""
    name = Path(name).name
    path = _dataset_memo_path(name, sft_dir=sft_dir)
    doc: dict[str, Any] = {
        "dataset": name,
        "memo": memo,
        "author": author or "anonymous",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return doc


def load_dataset_memo(name: str, *, sft_dir: Path = SFT_DIR) -> dict[str, Any]:
    """データセットのメモを読む（無い・壊れている場合は空の内容を返す）。"""
    name = Path(name).name
    path = _dataset_memo_path(name, sft_dir=sft_dir)
    empty: dict[str, Any] = {"dataset": name, "memo": "", "author": None, "updated_at": None}
    if not path.is_file():
        return empty
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    return doc if isinstance(doc, dict) else empty


def list_sft_datasets(*, sft_dir: Path = SFT_DIR) -> list[dict[str, Any]]:
    """data/sft 配下の JSONL 一覧（件数・メモ概要つき）。"""
    sft_dir = Path(sft_dir)
    if not sft_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(sft_dir.glob("*.jsonl")):
        try:
            # 件数は行数だけで済ませる（全文 parse しない）
            with path.open("r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
        except (OSError, UnicodeDecodeError):
            n = -1
        memo_preview = str(load_dataset_memo(path.name, sft_dir=sft_dir).get("memo") or "")[:80]
        rows.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "n": n,
                "memo_preview": memo_preview,
            }
        )
    return rows


def append_ui_train_history(
    entry: dict[str, Any],
    *,
    finetune_dir: Path = FINETUNE_DIR,
) -> Path:
    """UI / バッチ学習の設定履歴を追記する。"""
    finetune_dir = Path(finetune_dir)
    finetune_dir.mkdir(parents=True, exist_ok=True)
    path = finetune_dir / "ui_train_history.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def read_ui_train_history(*, finetune_dir: Path = FINETUNE_DIR, limit: int = 50) -> list[dict[str, Any]]:
    """直近の UI 学習設定履歴（新しい順）。"""
    path = Path(finetune_dir) / "ui_train_history.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(rows[-limit:]))


# ---------------------------------------------------------------------------
# 実験 ID（lab）とメモ
# ---------------------------------------------------------------------------


def get_vram_stats() -> dict[str, Any]:
    """CUDA VRAM 使用量を1回取得する（ポーリングしない用途向け）。"""
    try:
        import torch
    except ImportError:
        return {"ok": False, "error": "torch がインストールされていません"}
    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA が利用できません（CPU 版 torch の可能性）"}
    device = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    used_b = total_b - free_b
    return {
        "ok": True,
        "device": device,
        "name": torch.cuda.get_device_name(device),
        "total_gb": round(total_b / (1024**3), 2),
        "used_gb": round(used_b / (1024**3), 2),
        "free_gb": round(free_b / (1024**3), 2),
        "allocated_gb": round(allocated / (1024**3), 2),
        "reserved_gb": round(reserved / (1024**3), 2),
        "used_pct": round(100.0 * used_b / total_b, 1) if total_b else None,
    }


def release_vram() -> dict[str, Any]:
    """Python 参照の掃除 + CUDA キャッシュ解放を試み、解放後の統計を返す。

    注意: 他プロセスや、まだ参照が残っているモデルのメモリは完全には消えません。
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        return {"ok": False, "error": "torch がインストールされていません", "after": None}
    after = get_vram_stats()
    return {"ok": bool(after.get("ok")), "after": after, "error": after.get("error")}


def make_lab_id(name: str) -> str:
    """実験 ID 用スラッグ。"""
    return _safe_slug(name) or "lab"


def _lab_path(lab_id: str, *, finetune_dir: Path = FINETUNE_DIR) -> Path:
    return Path(finetune_dir) / "labs" / make_lab_id(lab_id) / "lab.json"


def create_lab(
    lab_id: str,
    *,
    title: str = "",
    shared_memo: str = "",
    author: str = "",
    finetune_dir: Path = FINETUNE_DIR,
) -> dict[str, Any]:
    """実験 ID を新規作成（既存なら読み込んで返す）。"""
    lid = make_lab_id(lab_id)
    path = _lab_path(lid, finetune_dir=finetune_dir)
    if path.is_file():
        return load_lab(lid, finetune_dir=finetune_dir)
    now = datetime.now().isoformat(timespec="seconds")
    lab: dict[str, Any] = {
        "lab_id": lid,
        "title": title or lid,
        "created_at": now,
        "updated_at": now,
        "shared_memo": shared_memo,
        "shared_log": [],
        "run_ids": [],
    }
    if shared_memo.strip():
        lab["shared_log"].append(
            {
                "at": now,
                "author": author or "anonymous",
                "action": "create",
                "memo": shared_memo,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lab


def load_lab(lab_id: str, *, finetune_dir: Path = FINETUNE_DIR) -> dict[str, Any]:
    """実験 ID の lab.json を読む。"""
    path = _lab_path(lab_id, finetune_dir=finetune_dir)
    if not path.is_file():
        raise FileNotFoundError(f"lab がありません: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_labs(*, finetune_dir: Path = FINETUNE_DIR) -> list[dict[str, Any]]:
    """実験 ID 一覧（新しい順に近い・ディレクトリ名順）。"""
    root = Path(finetune_dir) / "labs"
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/lab.json")):
        try:
            lab = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "lab_id": lab.get("lab_id") or path.parent.name,
                "title": lab.get("title"),
                "updated_at": lab.get("updated_at"),
                "n_runs": len(lab.get("run_ids") or []),
                "shared_memo_preview": str(lab.get("shared_memo") or "")[:80],
            }
        )
    rows.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
    return rows


def save_shared_memo(
    lab_id: str,
    memo: str,
    *,
    author: str = "",
    title: str | None = None,
    finetune_dir: Path = FINETUNE_DIR,
) -> dict[str, Any]:
    """共同メモを保存し、履歴に1件追記する。"""
    lid = make_lab_id(lab_id)
    path = _lab_path(lid, finetune_dir=finetune_dir)
    if path.is_file():
        lab = load_lab(lid, finetune_dir=finetune_dir)
    else:
        lab = create_lab(lid, shared_memo=memo, author=author, finetune_dir=finetune_dir)
        return lab
    now = datetime.now().isoformat(timespec="seconds")
    lab["shared_memo"] = memo
    lab["updated_at"] = now
    if title is not None:
        lab["title"] = title
    log = list(lab.get("shared_log") or [])
    log.append(
        {
            "at": now,
            "author": author or "anonymous",
            "action": "update",
            "memo": memo,
        }
    )
    lab["shared_log"] = log[-100:]  # 直近100件まで
    path.write_text(json.dumps(lab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lab


def link_run_to_lab(
    lab_id: str,
    run_id: str,
    *,
    finetune_dir: Path = FINETUNE_DIR,
) -> dict[str, Any]:
    """run_id を実験 ID に紐づける（lab が無ければ作成）。"""
    lid = make_lab_id(lab_id)
    path = _lab_path(lid, finetune_dir=finetune_dir)
    if not path.is_file():
        create_lab(lid, finetune_dir=finetune_dir)
    lab = load_lab(lid, finetune_dir=finetune_dir)
    run_ids = list(lab.get("run_ids") or [])
    if run_id not in run_ids:
        run_ids.append(run_id)
    lab["run_ids"] = run_ids
    lab["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(lab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lab


def save_run_memo(
    run_id: str,
    memo: str,
    *,
    lab_id: str | None = None,
    author: str = "",
    finetune_dir: Path = FINETUNE_DIR,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """各実験（run）のメモを result.json / memo.json / index に保存する。"""
    finetune_dir = Path(finetune_dir)
    result = load_run(run_id, finetune_dir=finetune_dir)
    now = datetime.now().isoformat(timespec="seconds")
    if lab_id:
        result["lab_id"] = make_lab_id(lab_id)
        link_run_to_lab(result["lab_id"], run_id, finetune_dir=finetune_dir)
    result["memo"] = memo
    result["memo_updated_at"] = now
    result["memo_author"] = author or result.get("memo_author") or "anonymous"
    save_run(result, work_dir=work_dir, finetune_dir=finetune_dir)

    memo_doc = {
        "run_id": run_id,
        "lab_id": result.get("lab_id"),
        "memo": memo,
        "author": result["memo_author"],
        "updated_at": now,
    }
    memo_path = finetune_dir / run_id / "memo.json"
    memo_path.write_text(json.dumps(memo_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def load_run_memo(run_id: str, *, finetune_dir: Path = FINETUNE_DIR) -> dict[str, Any]:
    """run のメモを読む（memo.json 優先、無ければ result.json）。"""
    finetune_dir = Path(finetune_dir)
    memo_path = finetune_dir / run_id / "memo.json"
    if memo_path.is_file():
        return json.loads(memo_path.read_text(encoding="utf-8"))
    result = load_run(run_id, finetune_dir=finetune_dir)
    return {
        "run_id": run_id,
        "lab_id": result.get("lab_id"),
        "memo": result.get("memo") or "",
        "author": result.get("memo_author"),
        "updated_at": result.get("memo_updated_at"),
    }


def save_common_memo(
    memo: str,
    *,
    author: str = "",
    finetune_dir: Path = FINETUNE_DIR,
) -> dict[str, Any]:
    """全実験共通のメモを保存し、履歴に1件追記する（共同メモと同じ log 形式）。"""
    finetune_dir = Path(finetune_dir)
    path = finetune_dir / "common_memo.json"
    doc: dict[str, Any] = {"memo": "", "author": None, "updated_at": None, "log": []}
    existed = path.is_file()
    if existed:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                doc.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    now = datetime.now().isoformat(timespec="seconds")
    log = list(doc.get("log") or [])
    log.append(
        {
            "at": now,
            "author": author or "anonymous",
            "action": "update" if existed else "create",
            "memo": memo,
        }
    )
    doc["memo"] = memo
    doc["author"] = author or doc.get("author") or "anonymous"
    doc["updated_at"] = now
    doc["log"] = log[-100:]  # 直近100件まで
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return doc


def load_common_memo(*, finetune_dir: Path = FINETUNE_DIR) -> dict[str, Any]:
    """共通メモを読む（無ければ空の内容を返す）。"""
    path = Path(finetune_dir) / "common_memo.json"
    if not path.is_file():
        return {"memo": "", "author": None, "updated_at": None, "log": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _split_messages_records(
    records: list[dict[str, Any]],
    *,
    valid_ratio: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """messages JSONL を train / valid に分割する。"""
    if not records:
        raise ValueError("学習データが空です")
    if len(records) == 1:
        return records, records
    n_valid = max(1, int(round(len(records) * valid_ratio)))
    n_valid = min(n_valid, len(records) - 1)
    return records[:-n_valid], records[-n_valid:]


# ---------------------------------------------------------------------------
# 評価（train 用。torch は呼び出し時に import）
# ---------------------------------------------------------------------------


def check_format(text: str) -> tuple[bool, str, dict | None]:
    """JSON 形式チェック。"""
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_error: {type(e).__name__}", None
    if not isinstance(obj, dict):
        return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}:
        return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj:
        return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or not isinstance(obj["needs_review"], bool):
        return False, "wrong_type", obj
    return True, "", obj


def check_content(obj: dict | None, gold_entry: dict[str, Any]) -> bool:
    """内容チェック。"""
    if obj is None:
        return False
    if obj.get("needs_review") != gold_entry["needs_review"]:
        return False
    return all(w in obj.get("answer", "") for w in gold_entry["must_include"])


def evaluate_model(
    model: Any,
    tok: Any,
    test: list[dict[str, Any]],
    gold: dict[str, dict],
    config: dict[str, Any],
    label: str,
    *,
    work_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """test セットを採点する（notebook の evaluate 相当）。"""
    import torch

    def build_messages(example: dict[str, Any]) -> list[dict[str, str]]:
        user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def generate(example: dict[str, Any]) -> str:
        prompt = tok.apply_chat_template(
            build_messages(example),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tok.pad_token_id,
                **config["gen"],
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for ex in test:
        raw = generate(ex)
        ok_fmt, reason, obj = check_format(raw)
        loose = obj
        if not ok_fmt:
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                loose = json.loads(m.group(0)) if m else None
            except Exception:
                loose = None
        ok_content = check_content(loose if isinstance(loose, dict) else None, gold[ex["question"]])
        rows.append(
            {
                "question": ex["question"],
                "raw": raw,
                "format_ok": ok_fmt,
                "format_reason": reason,
                "content_ok": ok_content,
                "both_ok": ok_fmt and ok_content,
                "gold_needs_review": gold[ex["question"]]["needs_review"],
            }
        )
    n = len(rows)
    summary = {
        "label": label,
        "n": n,
        "format_rate": sum(r["format_ok"] for r in rows) / n,
        "content_rate": sum(r["content_ok"] for r in rows) / n,
        "both_rate": sum(r["both_ok"] for r in rows) / n,
        "seconds": round(time.time() - t0, 1),
    }
    if work_dir is not None:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        (work / f"eval_{label}.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary, rows


def train_lora(
    config: dict[str, Any] | None = None,
    *,
    work_dir: Path | None = None,
    run_note: str | None = None,
    skip_eval: bool = False,
    train_jsonl: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    lab_id: str | None = None,
    memo: str | None = None,
    memo_author: str = "",
) -> dict[str, Any]:
    """LoRA 学習し、result.json を保存して返す。

    既定は FAQ 分割データ（評価あり）。`train_jsonl` を渡すと messages JSONL で学習し、
    FAQ 評価はスキップする（カスタムデータに gold が無いため）。
    `lab_id` / `memo` を渡すと実験 ID 紐づけと run メモを保存する
    （`memo_author` はメモの記録者名）。

    transformers / peft / trl / datasets / torch が必要（lazy import）。
    """
    import torch
    import transformers
    import peft as peft_mod
    import trl as trl_mod
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    def _progress(stage: str, progress: float | None = None, message: str = "", **extra: Any) -> None:
        if progress_callback is None:
            return
        payload: dict[str, Any] = {"stage": stage, "progress": progress, "message": message}
        payload.update(extra)
        progress_callback(payload)

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if run_note is not None:
        cfg["run_note"] = run_note
    if lab_id:
        cfg["lab_id"] = make_lab_id(lab_id)
    if memo is not None:
        cfg["memo"] = memo
    work = Path(work_dir) if work_dir is not None else REPO_ROOT / "notebooks" / "qlora_lab"
    work.mkdir(parents=True, exist_ok=True)
    cfg["work_dir"] = str(work)

    use_custom = train_jsonl is not None
    test: list[dict[str, Any]] = []
    gold: dict[str, dict] = {}
    if use_custom:
        jsonl_path = Path(train_jsonl)
        records = read_jsonl(jsonl_path)
        if not records:
            raise ValueError(f"学習 JSONL が空です: {jsonl_path}")
        for i, rec in enumerate(records):
            if "messages" not in rec:
                raise ValueError(f"{jsonl_path}:{i+1} に messages がありません")
        train_msgs, valid_msgs = _split_messages_records(records)
        write_jsonl(train_msgs, work / "train.jsonl")
        write_jsonl(valid_msgs, work / "valid.jsonl")
        cfg["dataset"] = str(jsonl_path.resolve())
        cfg["dataset_n"] = len(records)
        # カスタムデータは FAQ gold 評価ができない
        skip_eval = True
        train_ds = Dataset.from_list(train_msgs)
        valid_ds = Dataset.from_list(valid_msgs)
    else:
        train, valid, test, gold = build_faq_splits()
        (work / "train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8"
        )
        (work / "test.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in test), encoding="utf-8"
        )
        (work / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
        cfg["dataset"] = "builtin_faq"
        cfg["dataset_n"] = len(train)

        def to_chat(example: dict[str, Any]) -> dict[str, Any]:
            user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": example["target"]},
                ]
            }

        train_ds = Dataset.from_list([to_chat(e) for e in train])
        valid_ds = Dataset.from_list([to_chat(e) for e in valid])

    _progress("load", 0.05, "モデルを読み込み中…")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], revision=cfg["revision"], dtype=torch.float32
    ).to(device)
    model.config.use_cache = True

    before_summary: dict[str, Any]
    before_rows: list[dict[str, Any]]
    if skip_eval:
        before_summary = {
            "label": "before",
            "n": 0,
            "format_rate": None,
            "content_rate": None,
            "both_rate": None,
            "seconds": 0,
        }
        before_rows = []
    else:
        _progress("before_eval", 0.15, "学習前評価中…")
        before_summary, before_rows = evaluate_model(
            model, tok, test, gold, cfg, "before", work_dir=work
        )

    model.config.use_cache = False
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    sft = SFTConfig(
        output_dir=str(work / "ckpt"),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["warmup_steps"],
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        fp16=False,
        max_length=cfg["max_seq_len"],
        packing=False,
        report_to="none",
        seed=cfg["seed"],
    )

    class _ProgressCallback(TrainerCallback):
        """Trainer のログを progress_callback に流す。"""

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            if not state.max_steps:
                return
            frac = min(1.0, float(state.global_step) / float(state.max_steps))
            # 学習フェーズは全体の 0.2〜0.85 に割り当てる
            _progress(
                "train",
                0.2 + 0.65 * frac,
                f"学習中 step {state.global_step}/{state.max_steps}",
                global_step=state.global_step,
                max_steps=state.max_steps,
                logs=logs or {},
            )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        peft_config=lora,
        processing_class=tok,
        callbacks=[_ProgressCallback()],
    )
    trainer.model.print_trainable_parameters()
    _progress("train", 0.2, "学習開始…")
    t0 = time.time()
    trainer.train()
    training_seconds = round(time.time() - t0, 1)
    last_loss = extract_last_loss(trainer.state.log_history)
    print(
        f"学習時間: {training_seconds:.0f} 秒 | ステップ数: {trainer.state.global_step} "
        f"| エポック: {trainer.state.epoch:.1f} | last_loss: {last_loss}"
    )
    adapter_dir = work / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()

    if skip_eval:
        after_summary = {
            "label": "after",
            "n": 0,
            "format_rate": None,
            "content_rate": None,
            "both_rate": None,
            "seconds": 0,
        }
        after_rows: list[dict[str, Any]] = []
        regressed: list[str] = []
        knowledge_rows: list[dict[str, Any]] = []
    else:
        _progress("after_eval", 0.9, "学習後評価中…")
        after_summary, after_rows = evaluate_model(
            tuned, tok, test, gold, cfg, "after", work_dir=work
        )
        regressed = [
            b["question"]
            for b, a in zip(before_rows, after_rows)
            if b["both_ok"] and not a["both_ok"]
        ]
        knowledge_rows = []  # CLI では知識プローブは省略（ノートブックで実施）

    train_info = {
        "last_loss": last_loss,
        "training_seconds": training_seconds,
        "global_step": int(trainer.state.global_step),
        "epoch": float(trainer.state.epoch),
    }
    versions = {
        "transformers": transformers.__version__,
        "peft": peft_mod.__version__,
        "trl": trl_mod.__version__,
        "torch": torch.__version__,
    }
    _progress("save", 0.95, "result.json を保存中…")
    result = build_result(
        config=cfg,
        before=before_summary,
        after=after_summary,
        train_info=train_info,
        regressed=regressed,
        knowledge=knowledge_rows,
        versions=versions,
        run_note=str(cfg.get("run_note", "baseline")),
        lab_id=cfg.get("lab_id"),
        memo=str(cfg.get("memo") or ""),
    )
    if result.get("memo") and memo_author:
        result["memo_author"] = memo_author
    run_dir = save_run(result, work_dir=work, copy_eval_from=work)
    rid = str(result.get("run_id"))
    if result.get("lab_id"):
        link_run_to_lab(str(result["lab_id"]), rid)
    if result.get("memo"):
        (run_dir / "memo.json").write_text(
            json.dumps(
                {
                    "run_id": rid,
                    "lab_id": result.get("lab_id"),
                    "memo": result.get("memo"),
                    "author": result.get("memo_author") or "train",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print("保存先:", run_dir)
    _progress("done", 1.0, f"完了: {run_dir.name}", run_id=result.get("run_id"))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_make_dataset(args: argparse.Namespace) -> int:
    out = make_dataset(args.source, Path(args.output), glossary_path=Path(args.glossary))
    n = sum(1 for _ in out.read_text(encoding="utf-8").splitlines() if _.strip())
    print(f"wrote {n} records -> {out.resolve()}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    work = Path(args.work_dir) if args.work_dir else None
    rows = list_runs(finetune_dir=Path(args.finetune_dir), work_dir=work)
    if not rows:
        print("実験結果がありません（outputs/finetune/）。")
        return 0
    # 表形式（pandas が無ければ JSON）
    try:
        import pandas as pd

        print(pd.DataFrame(rows).to_string(index=False))
    except ImportError:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    work = Path(args.work_dir) if args.work_dir else None
    finetune_dir = Path(args.finetune_dir)
    rows = list_runs(finetune_dir=finetune_dir, work_dir=work)
    if args.run_ids:
        wanted = set(args.run_ids)
        rows = [r for r in rows if r.get("run_id") in wanted]
    if not rows:
        print("比較対象がありません。")
        return 1
    cols = [
        "run_id",
        "run_note",
        "lora_r",
        "lr",
        "num_train_epochs",
        "format_rate",
        "content_rate",
        "last_loss",
        "training_seconds",
    ]
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        show = [c for c in cols if c in df.columns]
        print(df[show].to_string(index=False))
    except ImportError:
        print(json.dumps([{k: r.get(k) for k in cols} for r in rows], ensure_ascii=False, indent=2))

    # config 差分
    results = []
    for r in rows:
        rid = r.get("run_id")
        if not rid:
            continue
        path = finetune_dir / str(rid) / "result.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    diffs = compare_config_diffs(results)
    if diffs:
        print("\nconfig の差分キー:")
        print(json.dumps(diffs, ensure_ascii=False, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = dict(DEFAULT_CONFIG)
    if args.lora_r is not None:
        cfg["lora_r"] = args.lora_r
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.epochs is not None:
        cfg["num_train_epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    note = args.note or cfg.get("run_note", "baseline")
    work = Path(args.work_dir) if args.work_dir else REPO_ROOT / "notebooks" / "qlora_lab"
    train_jsonl = Path(args.train_jsonl) if args.train_jsonl else None

    def _cli_progress(ev: dict[str, Any]) -> None:
        msg = ev.get("message") or ev.get("stage")
        if msg:
            print(f"[{ev.get('stage')}] {msg}")

    try:
        train_lora(
            cfg,
            work_dir=work,
            run_note=note,
            skip_eval=args.skip_eval,
            train_jsonl=train_jsonl,
            progress_callback=_cli_progress,
        )
    except ImportError as e:
        print(
            "学習に必要なパッケージがありません。\n"
            "  uv sync --group finetune\n"
            "  （torch は CUDA 版を別途入れてください）\n"
            f"詳細: {e}"
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI パーサ。"""
    p = argparse.ArgumentParser(description="LoRA ファインチューニングと実験管理")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("make-dataset", help="SFT 用 JSONL を生成")
    m.add_argument("--source", choices=("glossary", "faq"), required=True)
    m.add_argument("-o", "--output", required=True, help="出力 JSONL パス")
    m.add_argument("--glossary", default=str(GLOSSARY_PATH), help="glossary.json のパス")
    m.set_defaults(func=_cmd_make_dataset)

    l = sub.add_parser("list", help="実験一覧")
    l.add_argument("--finetune-dir", default=str(FINETUNE_DIR))
    l.add_argument("--work-dir", default=None)
    l.set_defaults(func=_cmd_list)

    c = sub.add_parser("compare", help="実験の比較表")
    c.add_argument("run_ids", nargs="*", help="比較する run_id（省略時は全部）")
    c.add_argument("--finetune-dir", default=str(FINETUNE_DIR))
    c.add_argument("--work-dir", default=None)
    c.set_defaults(func=_cmd_compare)

    t = sub.add_parser("train", help="FAQ または messages JSONL で LoRA 学習")
    t.add_argument("--note", default="baseline", help="実験メモ（run_id に入る）")
    t.add_argument("--lora-r", type=int, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--work-dir", default=None)
    t.add_argument(
        "--train-jsonl",
        default=None,
        help="messages 形式 JSONL（指定時は FAQ 評価をスキップ）",
    )
    t.add_argument("--skip-eval", action="store_true", help="学習前後評価をスキップ（デバッグ用）")
    t.set_defaults(func=_cmd_train)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
