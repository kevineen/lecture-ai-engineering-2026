"""モデル公開日（リリース日）の照合。

優先順:
1. 公式ソース（OpenRouter /models の created、Hugging Face Hub の createdAt）
2. 手元調査の CSV: data/localdata/model_data.csv
3. 不明（None / 表示は '(不明)'）

CSV は自分で調べた公開日を追記・更新する想定。
ネットワーク取得は短いタイムアウトで失敗しても構わない（オフラインでも CSV のみで動く）。
"""

from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

# リポジトリ直下 data/localdata/model_data.csv
_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DATA_CSV = _REPO_ROOT / "data" / "localdata" / "model_data.csv"

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_HF_MODEL_API = "https://huggingface.co/api/models/{repo_id}"
_HTTP_TIMEOUT_SEC = 3.0

# CSV 列名の別名（日本語ヘッダ / 英語ヘッダ両対応）
_ID_COLUMNS = ("id", "model_id", "モデルid", "モデルID")
_NAME_COLUMNS = ("model名", "model_name", "name", "モデル名")
_DATE_COLUMNS = ("公開日", "released", "release_date", "released_at", "date")


def _normalize_model_id(model: str) -> str:
    """照合用にモデル名を正規化する（小文字・区切り統一・models/ 除去）。"""
    text = model.strip().lower()
    if text.startswith("models/"):
        text = text[len("models/") :]
    text = text.replace("_", "-")
    return text.strip()


def _aliases(model: str) -> list[str]:
    """完全一致用の候補キー一覧（org 付き / 無し）。"""
    norm = _normalize_model_id(model)
    keys = [norm]
    if "/" in norm:
        keys.append(norm.split("/", 1)[1])
    else:
        for org in ("openai", "google", "qwen", "anthropic", "meta-llama", "deepseek"):
            keys.append(f"{org}/{norm}")
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _pick_field(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    """行から最初に見つかった非空フィールドを返す。"""
    for key in candidates:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    # ヘッダの前後空白ずれにも一応対応
    normalized = {k.strip(): v for k, v in row.items() if k is not None}
    for key in candidates:
        if key in normalized and normalized[key] is not None:
            value = str(normalized[key]).strip()
            if value:
                return value
    return ""


def _parse_release_cell(raw: str) -> str | None:
    """公開日セルを YYYY-MM-DD に正規化。不正なら None。"""
    text = raw.strip()
    if not text:
        return None
    # 日付だけの場合
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            date.fromisoformat(text)
            return text
        except ValueError:
            return None
    # ISO datetime なら日付部分だけ
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            date.fromisoformat(text[:10])
            return text[:10]
        except ValueError:
            return None
    return None


@lru_cache(maxsize=1)
def _load_csv_release_index(csv_mtime_ns: int) -> dict[str, str]:
    """CSV から 正規化 id/名前 -> YYYY-MM-DD の索引を作る。

    csv_mtime_ns はキャッシュ破棄用（ファイル更新で再読込）。
    """
    del csv_mtime_ns  # キーに使うだけ
    path = MODEL_DATA_CSV
    if not path.is_file():
        return {}

    index: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            released = _parse_release_cell(_pick_field(row, _DATE_COLUMNS))
            if released is None:
                continue
            model_id = _pick_field(row, _ID_COLUMNS)
            model_name = _pick_field(row, _NAME_COLUMNS)
            for raw_key in (model_id, model_name):
                if not raw_key:
                    continue
                for alias in _aliases(raw_key):
                    # 先勝ち（CSV 上で先に書いた行を優先）
                    index.setdefault(alias, released)
    return index


def _csv_mtime_ns() -> int:
    try:
        return MODEL_DATA_CSV.stat().st_mtime_ns
    except OSError:
        return 0


def _from_csv(model: str) -> str | None:
    """手元の model_data.csv から公開日を探す。"""
    index = _load_csv_release_index(_csv_mtime_ns())
    if not index:
        return None

    for key in _aliases(model):
        if key in index:
            return index[key]

    # 部分一致（長いキー優先）: preview 付き ID など
    norm = _normalize_model_id(model)
    candidates = [(k, v) for k, v in index.items() if k in norm or norm in k]
    if not candidates:
        return None
    candidates.sort(key=lambda kv: len(kv[0]), reverse=True)
    return candidates[0][1]


def _http_get_json(url: str) -> Any | None:
    """短いタイムアウトで JSON を取得。失敗時は None。"""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "aie2026-ask/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


@lru_cache(maxsize=1)
def _openrouter_created_map() -> dict[str, str]:
    """OpenRouter の model_id -> YYYY-MM-DD。プロセス内キャッシュ。"""
    payload = _http_get_json(_OPENROUTER_MODELS_URL)
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if not isinstance(data, list):
        return {}
    out: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        created = item.get("created")
        if not isinstance(mid, str) or not isinstance(created, (int, float)):
            continue
        day = datetime.fromtimestamp(int(created), tz=timezone.utc).date().isoformat()
        out[_normalize_model_id(mid)] = day
    return out


def _from_openrouter(model: str) -> str | None:
    """公式（OpenRouter 掲載の created）から公開日相当を取る。"""
    mapping = _openrouter_created_map()
    if not mapping:
        return None
    for key in _aliases(model):
        if key in mapping:
            return mapping[key]
    return None


def _looks_like_hf_repo(model: str) -> bool:
    """Hugging Face の org/name 形式っぽいか。"""
    if model.count("/") != 1:
        return False
    org, name = model.split("/", 1)
    if not org or not name:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", org)) and bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", name)
    )


def _from_huggingface(model: str) -> str | None:
    """公式（HF Hub createdAt）から公開日相当を取る。"""
    if not _looks_like_hf_repo(model):
        return None
    payload = _http_get_json(_HF_MODEL_API.format(repo_id=model))
    if not isinstance(payload, dict):
        return None
    created = payload.get("createdAt") or payload.get("lastModified")
    if not isinstance(created, str) or len(created) < 10:
        return None
    return created[:10]


def lookup_release_date(model: str, *, provider: str | None = None) -> str | None:
    """モデル公開日を YYYY-MM-DD で返す。不明なら None。

    1) 公式 API（OpenRouter / Hugging Face）
    2) data/localdata/model_data.csv
    3) None（表示上は不明）
    """
    if not model or model in {"local-model", "default"}:
        return None

    # --- 1. 公式ソース ---
    if provider == "openrouter" or "/" in model:
        found = _from_openrouter(model)
        if found:
            return found

    found = _from_huggingface(model)
    if found:
        return found

    if provider != "openrouter":
        found = _from_openrouter(model)
        if found:
            return found

    # --- 2. 手元 CSV（公式が取れないときの調査データ） ---
    found = _from_csv(model)
    if found:
        return found

    # --- 3. 不明 ---
    return None


def format_release_date(model: str, *, provider: str | None = None) -> str:
    """表示用文字列。判明すれば日付、不明なら '(不明)'。"""
    day = lookup_release_date(model, provider=provider)
    return day if day is not None else "(不明)"


def release_date_as_date(model: str, *, provider: str | None = None) -> date | None:
    """テスト・比較用に date オブジェクトで返す。"""
    text = lookup_release_date(model, provider=provider)
    if text is None:
        return None
    return date.fromisoformat(text)


def clear_release_caches() -> None:
    """テスト用: CSV / OpenRouter のキャッシュを捨てる。"""
    _load_csv_release_index.cache_clear()
    _openrouter_created_map.cache_clear()
