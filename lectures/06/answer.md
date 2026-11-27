# 第6回 解答

## 演習

- STEP 1 のツールの名前：glossary_search_glossary
- STEP 2 の [tool call]：glossary_search_glossary({'query': '3レベル'})
- STEP 3 の [tool call]：fetch_fetch({'url': 'https://modelcontextprotocol.io/'})
- STEP 4 の [tool call]：aws_aws___search_documentation({'search_phrase': 'Amazon S3 バージョニング 概要', 'topics': ['general']})

- 気づき・疑問：MCPを経由するとglossary_search_glossaryになり、エージェント側のコードを変更しなくても、
設定にfetch awsなどを追加するだけでツールが増えるため楽ちんであった。ただし、信頼できるサーバーでないと使うのは気を付ける必要があると感じた。

## 宿題

- 取り組んだレベル：発展
- やったこと：
  - .cursor/mcp.jsonを作成
  - 同じ用語集サーバーを Cursor から利用
  - 調査URL　2026.11.27 時点
    https://cursor.com/docs/mcp
    https://cursor.com/help/customization/mcp

  - `.cursor/mcp.json` を用意して Agent モードでツール呼び出しを確認 (プロジェクト単位での設定)
  - `app/agent.py`（`--mcp --trace`）からの使い方と比較

- 結果：

| 設定を書く場所 |  (project単位) .cursor/mcp.json (WindowsPC丸ごと) OSドライブ:\Users\(user名)\.cursor\mcp.json |
| コマンドで起動する書き方 | `"command" + "args"` |
| URL で接続する書き方 | `"url"`　`"headers / OAuth"`なども追加可能 |
| 有効になる範囲 | 全体プロジェクト＋個別プロジェクト |

比較
設定置き場所：演習では app/mcp_servers.json、Cursorは .cursor/mcp.json
スキーマ：どちらも mcpServers / command + args / url と似ている
差分：cursorにはheaders/OAuth/ユーザー全体設定などがあり、チーム利用なども意識されている

CursorのstdioならCursorが主体で起動する


- 気づき・疑問：
　1.cursorについて
  Cursor では Run Mode の場合承認が出ない／ツール呼び出しが折りたたみで分かりづらく、コピーもできなかった。
  mcp_agent --trace のように流れを記録するには別の処理を組む必要がある。

　2.個別
　Ｑ＆Ａで同じ内容を２回チャットすれば性能が上がるような論文があったが、こういったツール系でも効果はあるのか（指示文？
  『Prompt Repetition Improves Non-Reasoning LLMs』
