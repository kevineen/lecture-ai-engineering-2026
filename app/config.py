"""講座全体で共通の設定。

APIキーやモデル名はコードに書かず、環境変数から読み込む。
"""

from __future__ import annotations

import os

# プロバイダ: gemini（既定） / openrouter
#   例: export AIE_PROVIDER=openrouter
DEFAULT_PROVIDER = "gemini"
PROVIDER = os.getenv("AIE_PROVIDER", DEFAULT_PROVIDER).strip().lower()

# 既定モデル。環境変数 AIE_MODEL を設定すると切り替えられる。
#   Gemini:     export AIE_MODEL=gemini-3.5-flash
#   OpenRouter: export AIE_MODEL=openai/gpt-4o-mini
DEFAULT_MODEL_GEMINI = "gemini-3.5-flash-lite"
DEFAULT_MODEL_OPENROUTER = "openai/gpt-4o-mini"


def _default_model_for(provider: str) -> str:
    if provider == "openrouter":
        return DEFAULT_MODEL_OPENROUTER
    return DEFAULT_MODEL_GEMINI


MODEL = os.getenv("AIE_MODEL") or _default_model_for(PROVIDER)

# temperature（未設定なら API 既定に任せる）。例: export AIE_TEMPERATURE=0.2
_raw_temperature = os.getenv("AIE_TEMPERATURE")
TEMPERATURE: float | None = float(_raw_temperature) if _raw_temperature not in (None, "") else None

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def api_error_message(status: int, detail: str = "") -> str:
    """HTTPステータスコードを、受講生向けの日本語の説明にする（ask.py と agent.py で共用）。"""
    if status == 429:
        return "レート制限に達しました。1分ほど待ってから再実行してください。"
    if status == 503:
        return "モデルが混み合っています（503）。少し待ってから再実行してください。"
    if status in (400, 401, 403, 404):
        return f"APIキーかモデル名に問題があります（{status}）: {detail}"
    return f"APIエラー（{status}）: {detail}"


def get_api_key() -> str:
    """環境変数 GEMINI_API_KEY を返す。未設定なら手順を表示して終了する。

    agent.py など既存コード向けの互換エイリアス。
    """
    return get_gemini_api_key()


def get_gemini_api_key() -> str:
    """環境変数 GEMINI_API_KEY を返す。未設定なら手順を表示して終了する。"""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "環境変数 GEMINI_API_KEY が見つかりません。\n"
            "1. GitHub の Settings → Codespaces → Secrets に GEMINI_API_KEY を登録し、\n"
            "   使う repository に自分の fork を選んでください。\n"
            "2. Codespaces を起動したあとに登録した場合は、Codespaces を再起動してください。\n"
            "   （左下の「Codespaces」→「Stop Current Codespace」→ 再度開く）"
        )
    return key


def get_openrouter_api_key() -> str:
    """環境変数 OPENROUTER_API_KEY を返す。未設定なら手順を表示して終了する。"""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(
            "環境変数 OPENROUTER_API_KEY が見つかりません。\n"
            "1. https://openrouter.ai/keys でキーを発行し、\n"
            "2. Codespaces Secrets またはローカルの環境変数に OPENROUTER_API_KEY を設定してください。\n"
            "   キーをコードやコミットに書かないでください。"
        )
    return key
