# 第2回 解答

## 演習

- 使ったモデル・設定（`result.json` の `config` から）：`Qwen/Qwen3.5-0.8B`、`lora_r=16`、`lora_alpha=32`、`lr=2e-4`、`num_train_epochs=20`、`seed=42`、`method=LoRA（量子化なし）`（run: `20261101_084642__baseline`、既定の講座 FAQ）
- 自分で学習を実行したか（`executed_training_myself`）：はい（手元 RTX 4090 で LoRA 学習）
- 学習前：`format_rate` = 0.095 / `content_rate` = 0.0
- 学習後：`format_rate` = 1.0 / `content_rate` = 0.857
- 退行した質問：なし
- 学習後に残った失敗と、その型（根拠なしの質問に断定した / 根拠ありの質問に確認要求で逃げた / 形式の崩れ）：形式は崩れていない。内容失敗は根拠なしへの断定寄りが残った（例: 文字数制限・フィードバック・オフィスアワー事前予約）
- 知識を問う質問（セクション 6）の結果（学習前の分類 → 学習後の分類）と気づいたこと：学習後は JSON 形式と「確認が必要」に寄る例が増えるが、断定が残る場合もある。形式が直っても知識保証にはならない
- 学習で変わったこと・変わらなかったこと：JSON 形式は短時間でも大きく改善する。内容（根拠の有無の判断）は改善するが完全ではない。loss の低さだけでは内容正解を保証しない

## 宿題

- 取り組んだレベル：発展
- 学習データ：講座用語集（`data/glossary.json`、用語 58 件）から `messages` 形式 JSONL を作成して学習に使う
  - コマンド：`uv run python -m app.finetune make-dataset --source glossary -o data/sft/glossary_sft.jsonl`
  - 件数：116 件（用語×質問テンプレ。生成済み: `data/sft/glossary_sft.jsonl`）
  - run_note / run_id：（TODO: 学習後に記入）
- 評価用の質問（自作。学習テンプレ「〜とは何ですか？」等とは別言い回し。詳細・gold は `data/sft/glossary_eval.json`）：

| 種類 | id | 質問 | 正解の要点（must_include） |
|---|---|---|---|
| 用語集にある語 | g1 | LoRA ではベースモデルの重みをどう扱い、実際に更新するのはどの部分ですか？ | 重みは**固定**、更新は**低ランク**行列だけ |
| 用語集にある語 | g2 | AIE2026 の宿題は何段階に分かれていて、どうすればその回は完了になりますか？ | **基礎**・**実装**・**発展**（どれか1つで完了） |
| 言い回しを変えた質問 | p1 | Parameter-Efficient Fine-Tuning を短く説明し、代表的な手法の名前を1つ挙げてください。 | ごく**一部**のパラメータを学習／代表例 **LoRA** |
| 言い回しを変えた質問 | p2 | Quantized LoRA は普通の LoRA と比べて、どの工夫でメモリを節約しますか？ | ベースを**量子化**して**メモリ**節約のうえ LoRA |
| 用語集にない語 | u1 | AIE2026 の最終課題の提出締切は何月何日ですか？ | 断定せず **確認**を求める（`needs_review` 相当） |
| 用語集にない語 | u2 | 第2回で使う Colab 無料枠の GPU は、1週間あたり何時間まで使えますか？ | 断定せず **確認**を求める（`needs_review` 相当） |

- 学習前後の比較：

| 指標 | 学習前 | 学習後 |
|---|---|---|
| content_rate（6問・must_include） | （TODO） | （TODO） |
| 観点メモ（未登録語で断定したか等） | （TODO） | （TODO） |
| last_loss | — | （TODO） |
| training_seconds | — | （TODO） |

具体出力の記録（学習前後で同じ質問。実行後に埋める）：

```text
Q: LoRA ではベースモデルの重みをどう扱い…（g1・用語集にある）
  before: …
  after: …
Q: AIE2026 の宿題は何段階…（g2・用語集にある）
  before: …
  after: …
Q: Parameter-Efficient Fine-Tuning を短く…（p1・言い換え）
  before: …
  after: …
Q: Quantized LoRA は普通の LoRA と比べて…（p2・言い換え）
  before: …
  after: …
Q: 最終課題の提出締切は…（u1・用語集にない）
  before: …
  after: …
Q: Colab 無料枠の GPU は…（u2・用語集にない）
  before: …
  after: …
```

- 用語集を検索する仕組み vs モデル学習のメリット・デメリット：（TODO。更新コスト・言い換え・根拠の明示・計算コストを、自分の before/after 結果に紐づけて書く）
- ファインチューニングが向く場面・向かない場面についての考察：（TODO）

---

### 作業メモ（提出前に消してよい）

1. `uv run python -m app.finetune make-dataset --source glossary -o data/sft/glossary_sft.jsonl`（済）
2. 評価用 Q / gold を `data/sft/glossary_eval.json` に用意し answer.md に転記（済・学習前）
3. ノートで FAQ 分割の代わりに JSONL を `Dataset.from_list` して学習（`CONFIG["run_note"]="glossary_sft"`）
4. 同じ評価6問で学習前後を記録 → content_rate / 具体出力 / 検索 vs 学習の考察を埋める
5. 実装版の控えは `lectures/02/answer_jissou.md`。提出用は本ファイル（発展）を `lec02` ブランチで出す想定
