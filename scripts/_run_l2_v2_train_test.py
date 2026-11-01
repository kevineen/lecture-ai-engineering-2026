"""notebooks/lora_format_tuning_my_l2_v2.ipynb の学習テストを一括実行する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "lora_format_tuning_my_l2_v2.ipynb"

# インストール（cell 2）・比較一覧（cell 21）・4bit 観察（cell 23）はスキップ
SKIP_CELLS = {2, 21, 23}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="l2_v2 ノートの学習〜result 保存")
    parser.add_argument("--epochs", type=int, default=None, help="CONFIG num_train_epochs を上書き")
    parser.add_argument("--lr", type=float, default=None, help="CONFIG lr を上書き")
    parser.add_argument("--note", default=None, help="CONFIG run_note を上書き")
    parser.add_argument("--dataset", default=None, help="CONFIG dataset を上書き")
    args = parser.parse_args(argv)

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}

    def display(obj):  # noqa: ANN001
        try:
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                print(obj.to_string(index=False))
                return
        except Exception:
            pass
        print(obj)

    ns["display"] = display
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code" or i in SKIP_CELLS:
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        print(f"\n===== cell {i} =====", flush=True)
        try:
            exec(compile(src, f"{NB_PATH.name}:cell{i}", "exec"), ns)
        except Exception as e:
            print(f"FAILED at cell {i}: {type(e).__name__}: {e}", flush=True)
            raise

        # CONFIG 確定後に上書き（cell 5）。fingerprint は cell 7 で付け直される。
        if i == 5:
            cfg = ns.get("CONFIG")
            if isinstance(cfg, dict):
                if args.epochs is not None:
                    cfg["num_train_epochs"] = args.epochs
                if args.lr is not None:
                    cfg["lr"] = args.lr
                if args.note is not None:
                    cfg["run_note"] = args.note
                if args.dataset is not None:
                    cfg["dataset"] = args.dataset
                print(
                    "CONFIG override:",
                    {k: cfg.get(k) for k in ("run_note", "dataset", "num_train_epochs", "lr")},
                )

    result = ns.get("result")
    if isinstance(result, dict):
        print("\n===== SUMMARY =====")
        print(
            json.dumps(
                {
                    "run_id": result.get("run_id"),
                    "run_note": result.get("run_note"),
                    "config": {
                        k: (result.get("config") or {}).get(k)
                        for k in (
                            "dataset",
                            "dataset_fingerprint",
                            "num_train_epochs",
                            "lora_r",
                            "lr",
                            "dataset_n",
                        )
                    },
                    "before": result.get("before"),
                    "after": result.get("after"),
                    "train": result.get("train"),
                    "regressed": result.get("regressed"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
