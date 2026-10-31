"""第2回 宿題（発展）: 用語集から SFT 用の Q&A データ（JSONL）を作る。

使い方:
    uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
    # または app.finetune 経由:
    uv run python -m app.finetune make-dataset --source glossary -o data/sft/glossary_sft.jsonl

できたファイルを Colab にアップロードし、LoRA 学習のデータとして使う。
質問の言い回しを増やす、別名でも質問を作る、検証用に一部を分ける、などを工夫してみよう。
"""

import json
import sys
from pathlib import Path

# `uv run python scripts/make_sft_dataset.py` でも import できるようにルートを追加
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.finetune import GLOSSARY_PATH, iter_glossary_sft_records


def main() -> None:
    for record in iter_glossary_sft_records(GLOSSARY_PATH):
        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
