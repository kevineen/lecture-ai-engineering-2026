"""app.finetune の実験管理・データセット（torch 不要）のテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from app.finetune import (
    append_ui_train_history,
    apply_dataset_meta,
    build_faq_splits,
    builtin_faq_editor_rows,
    faq_example_to_editor_row,
    build_result,
    check_glossary_content,
    compare_config_diffs,
    create_lab,
    editor_rows_to_glossary_eval,
    extract_last_loss,
    faq_example_to_messages,
    filter_runs,
    get_vram_stats,
    glossary_eval_to_editor_rows,
    infer_homework_level,
    iter_glossary_sft_records,
    link_run_to_lab,
    list_dataset_groups,
    list_labs,
    list_runs,
    list_sft_datasets,
    load_common_memo,
    load_dataset_memo,
    load_glossary_eval,
    load_lab,
    load_run_eval,
    load_run_memo,
    make_dataset,
    make_faq_dataset_meta,
    make_jsonl_dataset_meta,
    make_run_id,
    read_ui_train_history,
    release_vram,
    run_summary_row,
    save_common_memo,
    save_dataset_memo,
    save_glossary_eval,
    save_run,
    save_run_memo,
    save_shared_memo,
)


def test_extract_last_loss() -> None:
    assert extract_last_loss(None) is None
    assert extract_last_loss([]) is None
    assert extract_last_loss([{"loss": 1.5}, {"learning_rate": 1e-4}, {"loss": 0.02}]) == 0.02


def test_make_run_id() -> None:
    rid = make_run_id("lora_r=8")
    assert "__" in rid
    assert "lora_r" in rid


def test_build_and_save_run(tmp_path: Path) -> None:
    result = build_result(
        config={"lora_r": 16, "lr": 2e-4, "run_note": "baseline", "work_dir": "/tmp"},
        before={"label": "before", "n": 21, "format_rate": 0.1, "content_rate": 0.0, "both_rate": 0.0, "seconds": 1},
        after={"label": "after", "n": 21, "format_rate": 1.0, "content_rate": 0.9, "both_rate": 0.9, "seconds": 2},
        train_info={"last_loss": 0.01, "training_seconds": 100.0, "global_step": 40, "epoch": 20.0},
        run_note="baseline",
        run_id="20260101_000000__baseline",
    )
    assert "work_dir" not in result["config"]
    assert result["train"]["last_loss"] == 0.01

    finetune_dir = tmp_path / "finetune"
    work = tmp_path / "work"
    run_dir = save_run(result, work_dir=work, finetune_dir=finetune_dir)
    assert (run_dir / "result.json").is_file()
    assert (work / "result.json").is_file()
    assert (work / "runs" / result["run_id"] / "result.json").is_file()
    assert (finetune_dir / "index.jsonl").is_file()

    rows = list_runs(finetune_dir=finetune_dir, work_dir=work)
    assert len(rows) == 1
    assert rows[0]["format_rate"] == 1.0
    assert run_summary_row(result)["training_seconds"] == 100.0


def test_load_run_eval(tmp_path: Path) -> None:
    finetune_dir = tmp_path / "finetune"
    run_dir = finetune_dir / "20260101_000000__baseline"
    run_dir.mkdir(parents=True)
    doc = {
        "summary": {"label": "after", "n": 1, "format_rate": 1.0, "content_rate": 1.0, "both_rate": 1.0, "seconds": 1.0},
        "rows": [
            {
                "question": "提出期限はいつですか",
                "raw": '{"answer": "9 月 30 日", "needs_review": false}',
                "format_ok": True,
                "format_reason": "",
                "content_ok": True,
                "both_ok": True,
                "gold_needs_review": False,
            }
        ],
    }

    # 両方無し → None（評価スキップの run）
    assert load_run_eval("20260101_000000__baseline", finetune_dir=finetune_dir) is None

    # after のみ
    (run_dir / "eval_after.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    evals = load_run_eval("20260101_000000__baseline", finetune_dir=finetune_dir)
    assert evals is not None
    assert evals["before"] is None
    assert evals["after"]["rows"][0]["question"] == "提出期限はいつですか"

    # before も揃う
    before_doc = {**doc, "summary": {**doc["summary"], "label": "before", "format_rate": 0.0}}
    (run_dir / "eval_before.json").write_text(
        json.dumps(before_doc, ensure_ascii=False), encoding="utf-8"
    )
    evals = load_run_eval("20260101_000000__baseline", finetune_dir=finetune_dir)
    assert evals is not None
    assert evals["before"]["summary"]["label"] == "before"
    assert evals["after"]["summary"]["label"] == "after"


def test_compare_config_diffs() -> None:
    a = build_result(
        config={"lora_r": 16, "lr": 2e-4, "gen": {"do_sample": False}},
        before={},
        after={},
        run_id="a",
        run_note="a",
    )
    b = build_result(
        config={"lora_r": 8, "lr": 2e-4, "gen": {"do_sample": False}},
        before={},
        after={},
        run_id="b",
        run_note="b",
    )
    diffs = compare_config_diffs([a, b])
    assert "lora_r" in diffs
    assert "lr" not in diffs
    # nested dict があっても TypeError にならない
    assert "gen" not in diffs


def test_dataset_meta_fingerprint_and_filter(tmp_path: Path) -> None:
    train, valid, test, gold = build_faq_splits()
    meta_a = make_faq_dataset_meta(
        dataset_id="builtin_faq", faq={"k": "v"}, train=train, valid=valid, test=test, gold=gold
    )
    meta_b = make_faq_dataset_meta(
        dataset_id="builtin_faq",
        faq={"k": "changed"},
        train=train,
        valid=valid,
        test=test,
        gold=gold,
    )
    assert meta_a["dataset"] == "builtin_faq"
    assert meta_a["dataset_n"] == 14
    assert meta_a["dataset_fingerprint"] != meta_b["dataset_fingerprint"]

    cfg = apply_dataset_meta({"lora_r": 16}, meta_a)
    assert cfg["dataset_fingerprint"] == meta_a["dataset_fingerprint"]

    jsonl = tmp_path / "custom.jsonl"
    records = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
    jsonl.write_text(json.dumps(records[0], ensure_ascii=False) + "\n", encoding="utf-8")
    jmeta = make_jsonl_dataset_meta(jsonl, records)
    assert jmeta["dataset"] == "custom"
    assert "dataset_path" in jmeta
    assert "dataset_path" not in build_result(config=jmeta, before={}, after={}, run_id="x", run_note="x")["config"]

    a = build_result(
        config=apply_dataset_meta({"lora_r": 16}, meta_a),
        before={},
        after={"format_rate": 1.0, "content_rate": 0.9},
        run_id="run_a",
        run_note="a",
    )
    b = build_result(
        config=apply_dataset_meta({"lora_r": 8}, meta_b),
        before={},
        after={"format_rate": 1.0, "content_rate": 0.5},
        run_id="run_b",
        run_note="b",
    )
    rows = [run_summary_row(a), run_summary_row(b)]
    assert filter_runs(rows, dataset="builtin_faq") == rows
    assert len(filter_runs(rows, fingerprint=meta_a["dataset_fingerprint"])) == 1
    assert infer_homework_level({"dataset": "builtin_faq"}) == "基礎"
    assert infer_homework_level({"dataset": "glossary_sft"}) == "発展"
    assert infer_homework_level({"dataset": "my_faq", "homework_level": "実装"}) == "実装"
    assert len(filter_runs(rows, homework_level="基礎")) == 2
    groups = list_dataset_groups(rows)
    assert len(groups) == 2
    assert {g["dataset_fingerprint"] for g in groups} == {
        meta_a["dataset_fingerprint"],
        meta_b["dataset_fingerprint"],
    }


def test_faq_splits_and_messages() -> None:
    train, valid, test, gold = build_faq_splits()
    assert len(train) == 14
    assert len(valid) == 7
    assert len(test) == 21
    assert len(gold) == 21
    msg = faq_example_to_messages(train[0])
    assert msg["messages"][0]["role"] == "system"
    assert msg["messages"][-1]["role"] == "assistant"
    row = faq_example_to_editor_row(train[0])
    assert row["question"] == train[0]["question"]
    assert row["answer"]
    assert row["needs_review"] is False
    assert len(builtin_faq_editor_rows("train")) == 14
    assert len(builtin_faq_editor_rows("valid")) == 7
    assert len(builtin_faq_editor_rows("test")) == 21
    assert builtin_faq_editor_rows("train")[0].keys() >= {"question", "answer", "needs_review"}


def test_make_dataset_glossary(tmp_path: Path) -> None:
    out = tmp_path / "g.jsonl"
    make_dataset("glossary", out)
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(iter_glossary_sft_records())
    obj = json.loads(lines[0])
    assert obj["messages"][0]["role"] == "user"


def test_glossary_eval_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "glossary_eval.json"
    rows = [
        {
            "id": "g1",
            "kind": "in_glossary",
            "term_id": "lora",
            "question": "LoRA とは？",
            "gold_answer": "低ランクで固定した重みを保つ",
            "must_include": "固定, 低ランク",
            "needs_review": False,
        },
        {
            "id": "u1",
            "kind": "not_in_glossary",
            "term_id": "",
            "question": "締切は？",
            "gold_answer": "確認が必要",
            "must_include": "確認",
            "needs_review": True,
        },
    ]
    payload = editor_rows_to_glossary_eval(rows)
    assert len(payload["items"]) == 2
    assert payload["items"][0]["must_include"] == ["固定", "低ランク"]
    save_glossary_eval(payload, path)
    loaded = load_glossary_eval(path)
    back = glossary_eval_to_editor_rows(loaded)
    assert back[0]["question"] == "LoRA とは？"
    assert "低ランク" in back[0]["must_include"]
    assert check_glossary_content("重みは固定し低ランク行列だけ学ぶ", payload["items"][0])
    assert not check_glossary_content("わかりません", payload["items"][0])


def test_make_dataset_faq(tmp_path: Path) -> None:
    out = tmp_path / "f.jsonl"
    make_dataset("faq", out)
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 14


def test_list_sft_datasets_and_ui_history(tmp_path: Path) -> None:
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "a.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = list_sft_datasets(sft_dir=sft)
    assert len(rows) == 1
    assert rows[0]["name"] == "a.jsonl"
    assert rows[0]["n"] == 1
    assert rows[0]["memo_preview"] == ""  # メモ無し

    finetune_dir = tmp_path / "finetune"
    append_ui_train_history(
        {"batch_id": "b1", "queue": [{"run_note": "x", "lora_r": 8}]},
        finetune_dir=finetune_dir,
    )
    hist = read_ui_train_history(finetune_dir=finetune_dir)
    assert len(hist) == 1
    assert hist[0]["batch_id"] == "b1"


def test_vram_helpers_do_not_crash() -> None:
    stats = get_vram_stats()
    assert "ok" in stats
    released = release_vram()
    assert "ok" in released
    assert "after" in released


def test_lab_and_run_memo(tmp_path: Path) -> None:
    finetune_dir = tmp_path / "finetune"
    work = tmp_path / "work"
    lab = create_lab("lec02 sweep", title="HP sweep", shared_memo="初版", author="a", finetune_dir=finetune_dir)
    assert lab["lab_id"] == "lec02_sweep"
    save_shared_memo("lec02_sweep", "方針を更新", author="b", finetune_dir=finetune_dir)
    loaded = load_lab("lec02_sweep", finetune_dir=finetune_dir)
    assert "方針を更新" in loaded["shared_memo"]
    assert len(loaded["shared_log"]) >= 2
    assert list_labs(finetune_dir=finetune_dir)

    result = build_result(
        config={"lora_r": 8},
        before={},
        after={},
        run_id="20260101_000000__r8",
        run_note="r8",
        lab_id="lec02_sweep",
        memo="",
    )
    save_run(result, work_dir=work, finetune_dir=finetune_dir)
    link_run_to_lab("lec02_sweep", "20260101_000000__r8", finetune_dir=finetune_dir)
    save_run_memo(
        "20260101_000000__r8",
        "loss は下がったが content は同じ",
        lab_id="lec02_sweep",
        author="b",
        finetune_dir=finetune_dir,
        work_dir=work,
    )
    memo = load_run_memo("20260101_000000__r8", finetune_dir=finetune_dir)
    assert "content" in memo["memo"]
    lab2 = load_lab("lec02_sweep", finetune_dir=finetune_dir)
    assert "20260101_000000__r8" in lab2["run_ids"]
    rows = list_runs(finetune_dir=finetune_dir)
    assert rows[0].get("lab_id") == "lec02_sweep"
    assert rows[0].get("memo_preview")


def test_dataset_memo(tmp_path: Path) -> None:
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "a.jsonl").write_text("{}\n", encoding="utf-8")

    # メモ無し → 空の内容・空の preview
    empty = load_dataset_memo("a.jsonl", sft_dir=sft)
    assert empty["memo"] == ""
    assert list_sft_datasets(sft_dir=sft)[0]["memo_preview"] == ""

    save_dataset_memo("a.jsonl", "glossary 由来でノイズ気味", author="a", sft_dir=sft)
    loaded = load_dataset_memo("a.jsonl", sft_dir=sft)
    assert "ノイズ" in loaded["memo"]
    assert loaded["dataset"] == "a.jsonl"
    assert loaded["author"] == "a"
    assert loaded["updated_at"]
    # 一覧の memo_preview に反映される
    assert "ノイズ" in list_sft_datasets(sft_dir=sft)[0]["memo_preview"]


def test_common_memo(tmp_path: Path) -> None:
    finetune_dir = tmp_path / "finetune"

    # 未作成 → 空の内容
    empty = load_common_memo(finetune_dir=finetune_dir)
    assert empty["memo"] == ""
    assert empty["log"] == []

    first = save_common_memo("初版", author="a", finetune_dir=finetune_dir)
    assert first["log"][0]["action"] == "create"
    second = save_common_memo("方針を更新", author="b", finetune_dir=finetune_dir)
    assert second["log"][-1]["action"] == "update"
    assert len(second["log"]) == 2

    loaded = load_common_memo(finetune_dir=finetune_dir)
    assert loaded["memo"] == "方針を更新"
    assert loaded["author"] == "b"
    assert loaded["updated_at"]
