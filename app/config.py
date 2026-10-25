"""講座全体で共通の設定。

APIキーやモデル名はコードに書かず、環境変数から読み込む。
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

# プロバイダ: gemini（既定） / openrouter / local（およびエイリアス）
#   例: export AIE_PROVIDER=openrouter
#       export AIE_PROVIDER=local   # vLLM / SGLang / LM Studio（OpenAI 互換）
DEFAULT_PROVIDER = "gemini"
PROVIDER = os.getenv("AIE_PROVIDER", DEFAULT_PROVIDER).strip().lower()

# local 系のエイリアス（いずれも OpenAI 互換エンドポイント）
LOCAL_PROVIDER_ALIASES = frozenset({"local", "vllm", "sglang", "lmstudio", "lm-studio"})

# --type / --provider 用。エンジン名 → 既定ポート（OpenAI 互換の慣例）
#   vLLM=8000 / SGLang=30000 / LM Studio=1234
LOCAL_ENGINE_DEFAULT_PORTS: dict[str, int] = {
    "local": 8000,
    "vllm": 8000,
    "sglang": 30000,
    "lmstudio": 1234,
    "lm-studio": 1234,
}

# 既定モデル。環境変数 AIE_MODEL を設定すると切り替えられる。
#   Gemini:     export AIE_MODEL=gemini-3.5-flash
#   OpenRouter: export AIE_MODEL=openai/gpt-4o-mini
#   Local:      export AIE_MODEL=Qwen/Qwen2.5-7B-Instruct（サーバ側でロードした名前）
DEFAULT_MODEL_GEMINI = "gemini-3.5-flash-lite"
DEFAULT_MODEL_OPENROUTER = "openai/gpt-4o-mini"
DEFAULT_MODEL_LOCAL = "local-model"


def is_local_provider(provider: str) -> bool:
    """vLLM / SGLang / LM Studio など OpenAI 互換のローカル系プロバイダか。"""
    return provider.strip().lower() in LOCAL_PROVIDER_ALIASES


def default_port_for_provider(provider: str) -> int:
    """ローカル推論エンジンの既定ポート。不明なら vLLM と同じ 8000。"""
    key = provider.strip().lower()
    return LOCAL_ENGINE_DEFAULT_PORTS.get(key, 8000)


def _default_model_for(provider: str) -> str:
    if provider == "openrouter":
        return DEFAULT_MODEL_OPENROUTER
    if is_local_provider(provider):
        return DEFAULT_MODEL_LOCAL
    return DEFAULT_MODEL_GEMINI


MODEL = os.getenv("AIE_MODEL") or _default_model_for(PROVIDER)

# temperature（未設定なら API 既定に任せる）。例: export AIE_TEMPERATURE=0.2
_raw_temperature = os.getenv("AIE_TEMPERATURE")
TEMPERATURE: float | None = float(_raw_temperature) if _raw_temperature not in (None, "") else None

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# 自宅 LAN / Tailscale 上の OpenAI 互換サーバ（vLLM / SGLang / LM Studio）
#   例: export AIE_BASE_URL=http://100.64.0.12:8000/v1          # Tailscale + vLLM
#       export AIE_BASE_URL=http://192.168.1.10:30000/v1        # LAN + SGLang
#       export AIE_BASE_URL=http://100.64.0.12:1234/v1          # Tailscale + LM Studio
#       export AIE_HOST=192.168.1.15                            # ホストのみ（ポートは --type で自動）
# 既定ポート目安: vLLM=8000 / SGLang=30000 / LM Studio=1234
DEFAULT_LOCAL_HOST = "127.0.0.1"
DEFAULT_LOCAL_BASE_URL = f"http://{DEFAULT_LOCAL_HOST}:8000/v1"
LOCAL_BASE_URL = (
    os.getenv("AIE_BASE_URL")
    or os.getenv("LOCAL_BASE_URL")
    or DEFAULT_LOCAL_BASE_URL
).rstrip("/")


def _local_host_from_env() -> str:
    """接続先ホスト。AIE_HOST 優先。なければ AIE_BASE_URL / LOCAL_BASE_URL の hostname。"""
    host = (os.getenv("AIE_HOST") or "").strip()
    if host:
        return host
    # 実行時の環境変数も見る（ポートは捨ててホストだけ使う）
    raw = (
        os.getenv("AIE_BASE_URL")
        or os.getenv("LOCAL_BASE_URL")
        or LOCAL_BASE_URL
    ).strip()
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    return parsed.hostname or DEFAULT_LOCAL_HOST


def build_local_base_url(provider: str, *, host: str | None = None, port: int | None = None) -> str:
    """エンジン種別に応じた OpenAI 互換 base_url（末尾スラッシュなし）を組み立てる。"""
    resolved_host = (host or _local_host_from_env()).strip() or DEFAULT_LOCAL_HOST
    resolved_port = port if port is not None else default_port_for_provider(provider)
    return f"http://{resolved_host}:{resolved_port}/v1"


def resolve_local_base_url(provider: str, base_url: str | None = None) -> str:
    """local 系の実効 base_url を決める。

    優先順位:
      1. 明示の base_url（--base-url）→ そのまま
      2. --type / --provider が vllm / sglang / lmstudio など具体的なエンジン
         → ホストは AIE_HOST または AIE_BASE_URL から、ポートはエンジン既定
      3. provider=local（汎用）→ AIE_BASE_URL / LOCAL_BASE_URL をそのまま
    """
    if base_url:
        return base_url.rstrip("/")

    key = provider.strip().lower()
    # 具体エンジン名のときはポートを自動選択（誤った AIE_BASE_URL のポートを上書き）
    if key in LOCAL_ENGINE_DEFAULT_PORTS and key != "local":
        return build_local_base_url(key)

    # 汎用 local: 環境変数のフル URL を尊重
    return LOCAL_BASE_URL.rstrip("/")


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


def get_local_api_key() -> str:
    """ローカル OpenAI 互換サーバ用の API キーを返す。

    未設定でも接続できることが多い（vLLM / SGLang / LM Studio）。
    OpenAI クライアントは空文字を嫌うため、未設定時は "EMPTY" を返す。
    """
    key = os.getenv("AIE_API_KEY") or os.getenv("LOCAL_API_KEY")
    if key:
        return key
    return "EMPTY"
