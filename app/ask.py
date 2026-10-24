"""第1回: LLMに質問するスクリプト（対話・履歴・プロバイダ切替・実験保存）。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --chat
    uv run python -m app.ask --provider openrouter --model openai/gpt-4o-mini --temperature 0.2 --chat

環境変数:
    AIE_PROVIDER=gemini|openrouter
    AIE_MODEL=...
    AIE_TEMPERATURE=0.2
    GEMINI_API_KEY / OPENROUTER_API_KEY

実験ログは outputs/ に JSON で保存する（.gitignore 済み。git には上げない）。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import OpenAI
from openai import APIStatusError as OpenAIAPIStatusError

from app.config import (
    MODEL,
    OPENROUTER_BASE_URL,
    PROVIDER,
    TEMPERATURE,
    api_error_message,
    get_gemini_api_key,
    get_openrouter_api_key,
)

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# リポジトリ直下の outputs/（実験ログ。git 管理外）
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


@dataclass
class Turn:
    """会話の1ターン（ユーザーまたはアシスタント）。"""

    role: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_ms: int | None = None


@dataclass
class ExperimentSession:
    """1回の実験セッション（設定 + 会話履歴）。outputs/ に保存する。"""

    provider: str
    model: str
    temperature: float | None
    started_at: str
    ended_at: str | None = None
    turns: list[Turn] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "notes": self.notes,
            },
            "turns": [asdict(t) for t in self.turns],
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(text: str) -> str:
    """ファイル名用にモデル名などを安全な文字列にする。"""
    slug = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    return slug.strip("_")[:80] or "model"


def save_experiment(session: ExperimentSession, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """実験セッションを outputs/ に JSON 保存し、保存先パスを返す。"""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    session.ended_at = _utc_now_iso()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_part = "default" if session.temperature is None else f"t{session.temperature}"
    name = f"{stamp}__{session.provider}__{_safe_slug(session.model)}__{temp_part}.json"
    path = outputs_dir / name
    path.write_text(
        json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _print_usage(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    if prompt_tokens is not None:
        print(f"[入力トークン数] {prompt_tokens}")
    if completion_tokens is not None:
        print(f"[出力トークン数] {completion_tokens}")


def ask_gemini(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
) -> tuple[str, int | None, int | None]:
    """Gemini に履歴付きで問い合わせる。戻り値は (本文, 入力トークン, 出力トークン)。"""
    client = genai.Client(api_key=get_gemini_api_key())
    # Google 形式: user / model + parts
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    config = None
    if temperature is not None:
        config = genai_types.GenerateContentConfig(temperature=temperature)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count if usage is not None else None
    completion_tokens = usage.candidates_token_count if usage is not None else None
    return response.text or "", prompt_tokens, completion_tokens


def ask_openrouter(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
) -> tuple[str, int | None, int | None]:
    """OpenRouter（OpenAI 互換）に履歴付きで問い合わせる。"""
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=get_openrouter_api_key())
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage is not None else None
    completion_tokens = usage.completion_tokens if usage is not None else None
    return text, prompt_tokens, completion_tokens


def ask_once(
    messages: list[dict[str, str]],
    *,
    provider: str,
    model: str,
    temperature: float | None,
) -> tuple[str, int | None, int | None]:
    """プロバイダに応じて1回呼び出す。messages は OpenAI 形式（user/assistant）。"""
    if provider == "openrouter":
        return ask_openrouter(messages, model=model, temperature=temperature)
    if provider == "gemini":
        return ask_gemini(messages, model=model, temperature=temperature)
    raise SystemExit(f"未対応のプロバイダです: {provider}（gemini / openrouter）")


def run_turn(
    history: list[dict[str, str]],
    question: str,
    *,
    provider: str,
    model: str,
    temperature: float | None,
    session: ExperimentSession | None,
) -> str:
    """1往復して履歴と実験セッションを更新し、応答本文を返す。"""
    import time

    history.append({"role": "user", "content": question})
    if session is not None:
        session.turns.append(Turn(role="user", content=question))

    started = time.perf_counter()
    answer, prompt_tokens, completion_tokens = ask_once(
        history,
        provider=provider,
        model=model,
        temperature=temperature,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    history.append({"role": "assistant", "content": answer})
    if session is not None:
        session.turns.append(
            Turn(
                role="assistant",
                content=answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_ms=elapsed_ms,
            )
        )

    _print_usage(prompt_tokens, completion_tokens)
    print(f"[応答時間] {elapsed_ms} ms")
    return answer


def interactive_loop(
    *,
    provider: str,
    model: str,
    temperature: float | None,
    first_question: str | None,
    save: bool,
    notes: str,
) -> None:
    """終了するまで会話を続ける（履歴を保持）。"""
    history: list[dict[str, str]] = []
    session = ExperimentSession(
        provider=provider,
        model=model,
        temperature=temperature,
        started_at=_utc_now_iso(),
        notes=notes,
    )

    print(f"[provider] {provider}")
    print(f"[model] {model}")
    print(f"[temperature] {temperature if temperature is not None else '(API既定)'}")
    print(f"[save] {'outputs/ に保存' if save else 'しない'}")
    print("終了するには exit / quit / 空行を入力\n")

    pending = first_question

    try:
        while True:
            if pending is not None:
                question = pending
                pending = None
                print(f"あなた> {question}")
            else:
                try:
                    question = input("あなた> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n終了します")
                    break

            if not question or question.lower() in {"exit", "quit"}:
                print("終了します")
                break

            try:
                answer = run_turn(
                    history,
                    question,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    session=session,
                )
            except genai_errors.APIError as e:
                print(api_error_message(e.code, e.message))
                raise SystemExit(1) from e
            except OpenAIAPIStatusError as e:
                print(api_error_message(e.status_code, str(e)))
                raise SystemExit(1) from e

            print(f"モデル> {answer}\n")
    finally:
        if save and session.turns:
            path = save_experiment(session)
            print(f"[実験ログ] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLMに質問する（Gemini / OpenRouter、対話・実験保存対応）"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="質問文。省略時は対話モード（--chat と同じ）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="対話モード（履歴を保ち、停止するまで続ける）",
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "openrouter"),
        default=PROVIDER,
        help="API プロバイダ（既定: 環境変数 AIE_PROVIDER または gemini）",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="モデル名（既定: 環境変数 AIE_MODEL またはプロバイダ既定）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="temperature（既定: 環境変数 AIE_TEMPERATURE。未指定なら API 既定）",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="実験ログを outputs/ に保存する（既定: する。--no-save で無効）",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="実験メモ（outputs の meta.notes に保存）",
    )
    args = parser.parse_args()

    # 引数なし、または --chat → 対話。引数のみ → 1問（履歴なし相当の単発でもセッション保存可）
    chat_mode = args.chat or args.question is None
    first = args.question
    if not chat_mode and first is None:
        first = DEFAULT_QUESTION

    if chat_mode:
        interactive_loop(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            first_question=first,
            save=args.save,
            notes=args.notes,
        )
        return

    # 単発モード（従来どおり1問。比較用に outputs にも残せる）
    history: list[dict[str, str]] = []
    session = ExperimentSession(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        started_at=_utc_now_iso(),
        notes=args.notes,
    )
    print(f"[provider] {args.provider}")
    print(f"[model] {args.model}")
    print(f"[temperature] {args.temperature if args.temperature is not None else '(API既定)'}")
    print(f"[質問] {first}\n")
    try:
        answer = run_turn(
            history,
            first,  # type: ignore[arg-type]
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            session=session,
        )
        print(answer)
    except genai_errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1) from e
    except OpenAIAPIStatusError as e:
        print(api_error_message(e.status_code, str(e)))
        raise SystemExit(1) from e
    finally:
        if args.save and session.turns:
            path = save_experiment(session)
            print(f"\n[実験ログ] {path}")


if __name__ == "__main__":
    main()
