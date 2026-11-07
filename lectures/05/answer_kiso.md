# 第5回 解答

## 演習

- `--list-tools` で見えたツール名と、第3回（`--mcp` なし）との違い：
- `--mcp --trace` で見えたツール呼び出し（ツール名と引数）と、`requests` の回数：
- MCP サーバーに公開したツールと、その説明文：
- fetch サーバーを足したときのツール呼び出し（ツール名と引数）：

## 宿題

- 取り組んだレベル：基礎
- やったこと：
  - `uv run python -m app.mcp_server --http` でサーバー起動
  - `app/mcp_servers.json` の `glossary` を `{"url": "http://127.0.0.1:8000/mcp"}` に変更して同じ質問
  - stdio と Streamable HTTP の違い（起動の仕方・プロセス・ポート）を記録
  - ツールの docstring を書き換え、`--list-tools` と呼ばれ方の変化を比較
- 結果：
  - stdio：
  - Streamable HTTP：
  - docstring 変更前後の `--list-tools` / 呼ばれ方：
- 気づき・疑問：
