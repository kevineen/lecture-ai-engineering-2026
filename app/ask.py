"""第1回: LLMに質問するスクリプト（対話・履歴・プロバイダ切替・実験保存）。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --chat
    uv run python -m app.ask --provider openrouter --model openai/gpt-4o-mini --temperature 0.2 --chat

    # temperature を変えて揺らぎを比較（低いほど決定的、高いほど多様）
    uv run python -m app.ask --temperature 0 "同じ質問で決定的な返答"
    uv run python -m app.ask --temperature 1.0 "同じ質問で多様な返答"

    # 対話中は /temp でその場で変更できる（履歴は保持）
    #   /temp          … 現在値を表示
    #   /temp 0.2      … 0.2 に設定
    #   /temp default  … API 既定に戻す

    # 自宅 LAN / Tailscale 上の vLLM・SGLang・LM Studio（OpenAI 互換）
    # --type / --provider で既定ポートを自動選択（vllm=8000 / sglang=30000 / lmstudio=1234）
    # --model default / local-model / auto なら /v1/models から実名を自動取得
    uv run python -m app.ask --type vllm --model default --chat
    uv run python -m app.ask --type sglang --model auto --chat
    uv run python -m app.ask --provider lmstudio --chat
    # ホストだけ環境変数、ポートは --type で自動:
    #   set AIE_HOST=192.168.1.15
    #   uv run python -m app.ask --type vllm --chat
    # ポートも含めて明示したいときだけ --base-url を使う
    uv run python -m app.ask --type vllm --base-url http://100.64.0.12:8001/v1 --chat

環境変数:
    AIE_PROVIDER=gemini|openrouter|local|vllm|sglang|lmstudio
    AIE_MODEL=...
    AIE_TEMPERATURE=0.2
    AIE_HOST=192.168.1.15              # local 系のホスト（ポートは --type で自動）
    AIE_BASE_URL=http://host:port/v1   # local 系（LOCAL_BASE_URL でも可。ホスト抽出にも使う）
    AIE_API_KEY=...                    # local 系は任意（未設定なら EMPTY）
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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import OpenAI
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError

from app.config import (
    DEFAULT_MODEL_LOCAL,
    LOCAL_BASE_URL,
    LOCAL_ENGINE_DEFAULT_PORTS,
    MODEL,
    OPENROUTER_BASE_URL,
    PROVIDER,
    TEMPERATURE,
    api_error_message,
    get_gemini_api_key,
    get_local_api_key,
    get_openrouter_api_key,
    is_local_provider,
    resolve_local_base_url,
)
from app.model_releases import format_release_date, lookup_release_date

# CLI / 環境変数で使えるプロバイダ名
PROVIDER_CHOICES = (
    "gemini",
    "openrouter",
    "local",
    "vllm",
    "sglang",
    "lmstudio",
    "lm-studio",
)

# --type 用（ローカル推論エンジンのみ。ポート自動選択の対象）
LOCAL_TYPE_CHOICES = ("vllm", "sglang", "lmstudio", "lm-studio", "local")
# 定義漏れ防止（config 側のキーと一致させる）
assert set(LOCAL_TYPE_CHOICES) == set(LOCAL_ENGINE_DEFAULT_PORTS)

# サーバ未指定時の仮モデル名。/v1/models から実名へ自動解決する
LOCAL_MODEL_PLACEHOLDERS = frozenset(
    {
        DEFAULT_MODEL_LOCAL,
        "default",
        "auto",
        "",
    }
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
    # 出力トークン / 応答秒（実験ログ・表示用）
    tokens_per_sec: float | None = None
    # アシスタント応答時に使った temperature（対話中の /temp 変更をログに残す）
    temperature: float | None = None


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
    # local 系の接続先（再現用。クラウド系は None）
    base_url: str | None = None
    # モデル公開日 YYYY-MM-DD（不明なら None）
    model_released: str | None = None

    def to_dict(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "notes": self.notes,
        }
        if self.base_url is not None:
            meta["base_url"] = self.base_url
        if self.model_released is not None:
            meta["model_released"] = self.model_released
        return {
            "meta": meta,
            "turns": [asdict(t) for t in self.turns],
        }


def effective_base_url(provider: str, base_url: str | None) -> str | None:
    """実験ログ・表示用の実効 base_url（local 系のみ）。

    --type vllm 等ではエンジン既定ポートを自動適用する。
    """
    if not is_local_provider(provider):
        return None
    return resolve_local_base_url(provider, base_url)


def list_local_model_ids(base_url: str, *, api_key: str | None = None, timeout_s: float = 10.0) -> list[str]:
    """OpenAI 互換サーバの GET /v1/models からモデル id 一覧を返す。"""
    root = base_url.rstrip("/")
    # base_url が .../v1 でも .../v1/ でも models に届くようにする
    models_url = f"{root}/models" if root.endswith("/v1") else f"{root}/v1/models"
    key = api_key if api_key is not None else get_local_api_key()
    req = Request(
        models_url,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        method="GET",
    )
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — 自宅 LAN / Tailscale 向け
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") or []
    ids: list[str] = []
    for item in data:
        mid = item.get("id") if isinstance(item, dict) else None
        if isinstance(mid, str) and mid.strip():
            ids.append(mid.strip())
    return ids


def resolve_local_model_name(
    model: str,
    *,
    provider: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """仮モデル名（default / local-model / auto）なら /v1/models の先頭を使う。

    明示名はそのまま返す。取得失敗時は分かりやすいエラーで終了する。
    """
    name = (model or "").strip()
    if not is_local_provider(provider):
        return name
    if name not in LOCAL_MODEL_PLACEHOLDERS:
        return name

    url = resolve_local_base_url(provider, base_url)
    try:
        ids = list_local_model_ids(url, api_key=api_key)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        raise SystemExit(
            f"モデル名 '{name or DEFAULT_MODEL_LOCAL}' の自動解決に失敗しました: {e}\n"
            f"  base_url={url}\n"
            "  /v1/models に届くか確認するか、--model にサーバ上の実名を指定してください。"
        ) from e

    if not ids:
        raise SystemExit(
            f"サーバにモデルがありません（GET {url}/models）。\n"
            "  vLLM / SGLang でモデルをロードしてから再実行してください。"
        )

    chosen = ids[0]
    extra = f"（他 {len(ids) - 1} 件）" if len(ids) > 1 else ""
    print(
        f"[model] '{name or DEFAULT_MODEL_LOCAL}' → '{chosen}' に自動解決{extra}",
        flush=True,
    )
    if len(ids) > 1:
        print(f"  利用可能: {', '.join(ids)}", flush=True)
    return chosen


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


def calc_tokens_per_sec(
    completion_tokens: int | None,
    elapsed_ms: int,
) -> float | None:
    """出力トークン数と経過時間から tok/s を求める。計測不能なら None。"""
    if completion_tokens is None or completion_tokens < 0 or elapsed_ms <= 0:
        return None
    return completion_tokens / (elapsed_ms / 1000.0)


def format_tokens_per_sec(tokens_per_sec: float | None) -> str | None:
    """表示用の tok/s 文字列。算出不能なら None。"""
    if tokens_per_sec is None:
        return None
    return f"{tokens_per_sec:.1f} tok/s"


def format_temperature(temperature: float | None) -> str:
    """表示用の temperature 文字列。"""
    return "(API既定)" if temperature is None else str(temperature)


def parse_temperature_value(raw: str) -> float | None:
    """CLI / 対話コマンド用に temperature 文字列を解釈する。

    数値 → float、default/none/api（大小無視）→ None（API 既定）。
    解釈できない場合は ValueError。
    """
    text = raw.strip().lower()
    if text in {"default", "none", "api", "null"}:
        return None
    value = float(text)
    # 主要 API の一般的な範囲。外れ値は誤入力の可能性が高いので拒否する
    if not 0.0 <= value <= 2.0:
        raise ValueError("temperature は 0.0〜2.0 の範囲で指定してください")
    return value


def try_handle_chat_command(
    line: str,
    *,
    temperature: float | None,
    session: ExperimentSession,
) -> tuple[bool, float | None]:
    """対話中のスラッシュコマンドを処理する。

    戻り値: (処理したか, 新しい temperature)。
    未対応の行なら (False, temperature) のまま。
    """
    stripped = line.strip()
    if not stripped.startswith("/"):
        return False, temperature

    parts = stripped.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in {"/temp", "/temperature"}:
        if not arg:
            print(f"[temperature] {format_temperature(temperature)}")
            return True, temperature
        try:
            new_temp = parse_temperature_value(arg)
        except ValueError as e:
            print(f"temperature の指定が不正です: {e}")
            print("例: /temp 0.2  /temp 1.0  /temp default")
            return True, temperature
        session.temperature = new_temp
        print(f"[temperature] {format_temperature(temperature)} → {format_temperature(new_temp)}")
        return True, new_temp

    if cmd in {"/help", "/?"}:
        print(
            "コマンド:\n"
            "  /temp [値]   temperature の表示・変更（例: /temp 0.2 /temp default）\n"
            "  /help        このヘルプ\n"
            "  exit / quit  終了"
        )
        return True, temperature

    print(f"未知のコマンドです: {cmd}（/help で一覧）")
    return True, temperature


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


def ask_openai_compatible(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    base_url: str,
    api_key: str,
) -> tuple[str, int | None, int | None]:
    """OpenAI 互換 Chat Completions API に履歴付きで問い合わせる。

    OpenRouter・vLLM・SGLang・LM Studio など、/v1/chat/completions を持つサーバ向け。
    """
    # base_url は末尾スラッシュ無しを想定（openai SDK が path を結合する）
    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage is not None else None
    completion_tokens = usage.completion_tokens if usage is not None else None
    return text, prompt_tokens, completion_tokens


def ask_openrouter(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
) -> tuple[str, int | None, int | None]:
    """OpenRouter（OpenAI 互換）に履歴付きで問い合わせる。"""
    return ask_openai_compatible(
        messages,
        model=model,
        temperature=temperature,
        base_url=OPENROUTER_BASE_URL,
        api_key=get_openrouter_api_key(),
    )


def ask_local(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    base_url: str,
    api_key: str | None = None,
) -> tuple[str, int | None, int | None]:
    """自宅 LAN / Tailscale 上の vLLM・SGLang・LM Studio に問い合わせる。"""
    return ask_openai_compatible(
        messages,
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key if api_key is not None else get_local_api_key(),
    )


def ask_once(
    messages: list[dict[str, str]],
    *,
    provider: str,
    model: str,
    temperature: float | None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, int | None, int | None]:
    """プロバイダに応じて1回呼び出す。messages は OpenAI 形式（user/assistant）。"""
    if provider == "openrouter":
        return ask_openrouter(messages, model=model, temperature=temperature)
    if provider == "gemini":
        return ask_gemini(messages, model=model, temperature=temperature)
    if is_local_provider(provider):
        # --base-url 明示 > エンジン既定ポート（--type vllm 等）> LOCAL_BASE_URL
        url = resolve_local_base_url(provider, base_url)
        return ask_local(
            messages,
            model=model,
            temperature=temperature,
            base_url=url,
            api_key=api_key,
        )
    raise SystemExit(
        f"未対応のプロバイダです: {provider}"
        f"（{' / '.join(PROVIDER_CHOICES)}）"
    )


def run_turn(
    history: list[dict[str, str]],
    question: str,
    *,
    provider: str,
    model: str,
    temperature: float | None,
    session: ExperimentSession | None,
    base_url: str | None = None,
    api_key: str | None = None,
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
        base_url=base_url,
        api_key=api_key,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    # 出力スループット（completion トークン ÷ 応答秒）
    tokens_per_sec = calc_tokens_per_sec(completion_tokens, elapsed_ms)

    history.append({"role": "assistant", "content": answer})
    if session is not None:
        session.turns.append(
            Turn(
                role="assistant",
                content=answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_ms=elapsed_ms,
                tokens_per_sec=tokens_per_sec,
                temperature=temperature,
            )
        )

    _print_usage(prompt_tokens, completion_tokens)
    # 例: [応答時間] 3837 ms (134.7 tok/s)
    tps_label = format_tokens_per_sec(tokens_per_sec)
    if tps_label is not None:
        print(f"[応答時間] {elapsed_ms} ms ({tps_label})")
    else:
        print(f"[応答時間] {elapsed_ms} ms")
    print(f"[temperature] {format_temperature(temperature)}")
    return answer


def _print_session_header(
    *,
    provider: str,
    model: str,
    temperature: float | None,
    base_url: str | None,
    save: bool | None = None,
    model_released: str | None = None,
) -> None:
    """対話・単発共通の設定表示。"""
    print(f"[provider] {provider}")
    # モデル名と公開日を同じ行に表示（不明時は (不明)）
    released = model_released or format_release_date(model, provider=provider)
    print(f"[model] {model}  (公開日: {released})")
    print(f"[temperature] {format_temperature(temperature)}")
    # local 系だけ endpoint を明示（接続先の取り違え防止）
    if is_local_provider(provider):
        print(f"[base_url] {base_url or resolve_local_base_url(provider)}")
    if save is not None:
        print(f"[save] {'outputs/ に保存' if save else 'しない'}")


def interactive_loop(
    *,
    provider: str,
    model: str,
    temperature: float | None,
    first_question: str | None,
    save: bool,
    notes: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """終了するまで会話を続ける（履歴を保持）。/temp で temperature を途中変更可。"""
    history: list[dict[str, str]] = []
    # 対話中に /temp で書き換えるため、ローカル変数として保持
    current_temperature = temperature
    resolved_base = effective_base_url(provider, base_url)
    released = lookup_release_date(model, provider=provider)
    session = ExperimentSession(
        provider=provider,
        model=model,
        temperature=current_temperature,
        started_at=_utc_now_iso(),
        notes=notes,
        base_url=resolved_base,
        model_released=released,
    )

    _print_session_header(
        provider=provider,
        model=model,
        temperature=current_temperature,
        base_url=resolved_base,
        save=save,
        model_released=released,
    )
    print("終了: exit / quit / 空行 ｜ 設定変更: /temp 0.2 ｜ ヘルプ: /help\n")

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

            # /temp などで推論パラメータをその場で変える（履歴は消さない）
            handled, current_temperature = try_handle_chat_command(
                question,
                temperature=current_temperature,
                session=session,
            )
            if handled:
                continue

            try:
                answer = run_turn(
                    history,
                    question,
                    provider=provider,
                    model=model,
                    temperature=current_temperature,
                    session=session,
                    base_url=base_url,
                    api_key=api_key,
                )
            except genai_errors.APIError as e:
                print(api_error_message(e.code, e.message))
                raise SystemExit(1) from e
            except OpenAIAPIStatusError as e:
                print(api_error_message(e.status_code, str(e)))
                raise SystemExit(1) from e
            except (OpenAIAPIConnectionError, OSError) as e:
                # 接続拒否・タイムアウトなど（自宅サーバ未起動・Tailscale 未接続）
                print(
                    f"ローカルサーバに接続できません: {e}\n"
                    f"  base_url={resolved_base or base_url or LOCAL_BASE_URL}\n"
                    "  vLLM / SGLang / LM Studio が起動しているか、"
                    "Tailscale / LAN の疎通を確認してください。"
                )
                raise SystemExit(1) from e

            print(f"モデル> {answer}\n")
    finally:
        if save and session.turns:
            path = save_experiment(session)
            print(f"[実験ログ] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLMに質問する（Gemini / OpenRouter / ローカル OpenAI 互換、対話・実験保存対応）"
        )
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
        choices=PROVIDER_CHOICES,
        default=None,
        help=(
            "API プロバイダ（既定: 環境変数 AIE_PROVIDER または gemini）。"
            "local / vllm / sglang / lmstudio は自宅・Tailscale 上の OpenAI 互換サーバ向け。"
            "vllm / sglang / lmstudio 指定時は既定ポートを自動選択（--type と同じ）"
        ),
    )
    parser.add_argument(
        "--type",
        dest="engine_type",
        choices=LOCAL_TYPE_CHOICES,
        default=None,
        help=(
            "ローカル推論エンジン（vllm / sglang / lmstudio / local）。"
            "指定すると --provider より優先し、既定ポートを自動選択する "
            f"（vllm={LOCAL_ENGINE_DEFAULT_PORTS['vllm']} / "
            f"sglang={LOCAL_ENGINE_DEFAULT_PORTS['sglang']} / "
            f"lmstudio={LOCAL_ENGINE_DEFAULT_PORTS['lmstudio']}）。"
            "ホストは AIE_HOST または AIE_BASE_URL から取得"
        ),
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=(
            "モデル名（既定: 環境変数 AIE_MODEL またはプロバイダ既定）。"
            "local 系で default / local-model / auto のときは /v1/models から自動解決"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default=None,
        help=(
            "temperature 0.0〜2.0（低いほど決定的、高いほど多様）。"
            "既定: 環境変数 AIE_TEMPERATURE。未指定・default なら API 既定。"
            "対話中は /temp でも変更可"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI 互換 API のベース URL（local 系向け・明示指定時はポート自動選択を上書き）。"
            "例: http://100.x.x.x:8000/v1 。未指定なら --type の既定ポート + "
            f"AIE_HOST / AIE_BASE_URL（例: {LOCAL_BASE_URL}）"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "local 系の API キー（任意）。未指定なら AIE_API_KEY / LOCAL_API_KEY / EMPTY。"
            "キーをシェル履歴に残したくない場合は環境変数を推奨"
        ),
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

    # --type が最優先。なければ --provider、それもなければ環境変数 AIE_PROVIDER
    provider = (args.engine_type or args.provider or PROVIDER).strip().lower()

    # --temperature 未指定なら環境変数、指定ありならパース（default で API 既定に戻せる）
    if args.temperature is None:
        temperature = TEMPERATURE
    else:
        try:
            temperature = parse_temperature_value(args.temperature)
        except ValueError as e:
            raise SystemExit(f"--temperature が不正です: {e}") from e

    # local 系: default / local-model / auto なら /v1/models から実名を取得
    model = resolve_local_model_name(
        args.model,
        provider=provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    # 引数なし、または --chat → 対話。引数のみ → 1問（履歴なし相当の単発でもセッション保存可）
    chat_mode = args.chat or args.question is None
    first = args.question
    if not chat_mode and first is None:
        first = DEFAULT_QUESTION

    if chat_mode:
        interactive_loop(
            provider=provider,
            model=model,
            temperature=temperature,
            first_question=first,
            save=args.save,
            notes=args.notes,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        return

    # 単発モード（従来どおり1問。比較用に outputs にも残せる）
    history: list[dict[str, str]] = []
    resolved_base = effective_base_url(provider, args.base_url)
    released = lookup_release_date(model, provider=provider)
    session = ExperimentSession(
        provider=provider,
        model=model,
        temperature=temperature,
        started_at=_utc_now_iso(),
        notes=args.notes,
        base_url=resolved_base,
        model_released=released,
    )
    _print_session_header(
        provider=provider,
        model=model,
        temperature=temperature,
        base_url=resolved_base,
        model_released=released,
    )
    print(f"[質問] {first}\n")
    try:
        answer = run_turn(
            history,
            first,  # type: ignore[arg-type]
            provider=provider,
            model=model,
            temperature=temperature,
            session=session,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        print(answer)
    except genai_errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1) from e
    except OpenAIAPIStatusError as e:
        print(api_error_message(e.status_code, str(e)))
        raise SystemExit(1) from e
    except (OpenAIAPIConnectionError, OSError) as e:
        print(
            f"ローカルサーバに接続できません: {e}\n"
            f"  base_url={resolved_base or args.base_url or LOCAL_BASE_URL}\n"
            "  vLLM / SGLang / LM Studio が起動しているか、"
            "Tailscale / LAN の疎通を確認してください。"
        )
        raise SystemExit(1) from e
    finally:
        if args.save and session.turns:
            path = save_experiment(session)
            print(f"\n[実験ログ] {path}")


if __name__ == "__main__":
    main()
