# HopRAG 中文 RAG 工具 — 設計文件 (v2)

- **日期**：2026-06-17
- **狀態**：設計已通過,待寫實作計畫
- **前置**：建立在 v1(`2026-06-17-agentic-multihop-rag-design.md`)的 agentic 迴圈之上

---

## 1. 目標與成功標準

一個**繁體中文**的互動式 RAG 工具:使用者對**自己的 PDF 文件**即時提問,看到 **naive RAG vs agentic RAG 並排**的中文答案,並附上:

1. 每一邊的 **RAG-Triad 品質分數**(檢索相關性 / 忠實度 / 回答相關性,0–100)
2. 每一邊的**作答時間**
3. 答案引用的**來源**(哪份 PDF、第幾頁)
4. agentic 的**多跳推理 trace**

**成功標準**:使用者打一個中文問題 → 約 1–2 分鐘內看到兩邊答案 + 三項分數 + 時間 + 來源;naive 與 agentic 的**分數差異**讓「多跳的價值」一目了然(白話、非工程術語的介面)。

## 2. 已確認決策

| 面向 | 決策 | 理由 |
|---|---|---|
| 語言 | 介面與作答全繁體中文 | 使用者語料是繁中 PDF |
| 語料 | 使用者自己的 PDF(`AI-test/`,iPAS 教材) | 使用者明確要求用自己的文件 |
| 模式 | **即時**提問(非預先跑好的 static) | 使用者要看真實、即時的差異 |
| Embedding | **`BAAI/bge-m3`**(多語、強) | 中文檢索品質;離線;一次性建索引可接受其較重的成本 |
| 評分 | **RAG Triad**(reference-free,LLM-as-judge) | 查證後確認為業界標準(TruLens / RAGAS);使用者問題無標準答案,不能用 EM/F1 |
| 對照 | 保留 naive vs agentic **並排** | 分數差就是使用者要看的「具體差異」 |
| LLM | Claude CLI(沿用 v1 的 `ClaudeClient`,登入態、無金鑰) | 一致 |

## 3. 元件

沿用 v1 既有元件(`types`、`retriever`、`indexer`、`claude_client`、`agentic_rag`、`naive_rag`);新增/調整如下。

| 元件 | 職責 | 介面 |
|---|---|---|
| `ingest.py` | PDF→逐頁文字(PyMuPDF)→重疊字元視窗切塊;標題 = `檔名 p.頁` | `window_text`, `chunk_pages`, `load_pdf_chunks(folder)` |
| `prompts_zh.py` | 中文版 prompt builder(schema 與英文相同) | `decompose/hop/synthesize/verify_prompt` + 四個 schema |
| `NaiveRAG`/`AgenticRAG` | 新增可注入 `prompts_mod` 參數(預設英文,不破壞 v1 eval) | `__init__(..., prompts_mod=None)` |
| `rag_eval.py` | RAG Triad:`score_triad(claude, q, answer, contexts)` → 三分數 + overall | LLM-judge,reference-free |
| `rag_app.py` | Flask app:`/api/ask` 並行跑 naive‖agentic、各自評分、回傳時間+來源+trace | `create_app(retriever, claude_factory, id2chunk, sources)` |
| `scripts/rag_tool.py` | 載入 `AI-test/` PDF → bge-m3 建索引 → 起伺服器 | CLI runner |
| `demo/rag.html` | 繁中前端:提問框、並排兩欄、三條分數、時間、來源、多跳 trace | 單一靜態頁,fetch `/api/ask` |

**Embedding 取得**:沿用 `indexer.make_bge_embed_fn(model_name="BAAI/bge-m3")`。

## 4. 評分定義(RAG Triad)

對每個答案,用 Claude 當評審回傳 0–100:

- **context_relevance(檢索相關性)**:檢索到的內容整體與問題的相關程度。
- **groundedness(忠實度)**:答案有多少是被檢索內容支持的(幻覺越多越低)。
- **answer_relevance(回答相關性)**:答案多直接、完整地回答了問題。
- `overall` = 三者平均。

naive 與 agentic 各打一組;UI 凸顯差異。

## 5. 資料流(一次提問)

```
中文問題
 → 並行:
     naive  = 單次檢索(同一 bge-m3 索引)→ 中文作答          (1 次 Claude)
     agentic = 多跳迴圈(prompts_zh)→ 中文作答 + trace        (~4–8 次 Claude)
 → 並行:對兩個答案各跑 RAG-Triad 評審                          (2 次 Claude)
 → 回傳 {naive, agentic}:各含 answer / scores / elapsed_sec / cited[來源] / trace
 → 前端並排呈現,分數與時間量化差異
```

**並行化**:naive‖agentic 用 thread 並跑;兩個評審呼叫也並跑——降低 wall-clock(每次 `claude -p` 啟動成本高)。

## 6. 限制與緩解

- **延遲**:每題約 1–2 分鐘,因為每次 `claude -p` 都重啟整個 CLI runtime。緩解:並行化 + 介面顯示「處理中」進度與步驟。這是「Claude CLI 當模型」的架構本質,spec 誠實記載。
- **成本**:每題 ~7–9 次 Claude 呼叫;預設用便宜模型(haiku),UI 可選 sonnet/opus。

## 7. 錯誤處理

- CLI/模型錯誤 → `/api/ask` 回友善的 502 + 訊息(不丟堆疊)。
- schema 容忍 null / 省略欄位(沿用 v1)。
- 子程序輸出強制 UTF-8(沿用 v1 修正,避免 Windows cp950)。
- 空檢索 / 空答案不崩潰。

## 8. 測試

- **純邏輯 TDD**:`window_text`/`chunk_pages`(切塊)、`score_triad`(評分解析)、`prompts_zh`(中文 + 注入)——已完成,46 passed。
- **app**:`rag_app` 用 Flask `test_client` + fake retriever/claude(不連網、不打 CLI)驗證 `/api/ask` 結構、`/api/info`、錯誤處理。
- **實測**:用真實 `AI-test/` PDF 跑幾個問題,確認中文作答正確、分數合理、來源正確、伺服器可連。

## 9. 安全 / 隱私

- `AI-test/`(使用者版權 PDF)**加入 `.gitignore`,絕不提交到 public repo**。索引(`.lancedb/`)亦 gitignore。只在本機建索引與查詢。

## 10. 範圍(YAGNI)

- **做**:§3 全部 + 用 `AI-test/` 即時查詢 + RAG-Triad 評分 + 並排 + 時間 + 來源 + 繁中介面。
- **之後/不做**:英文 HotpotQA static demo 不再當門面(eval harness 與既有 demo 留在 repo);不做使用者上傳檔案 UI(直接讀資料夾);不做帳號 / 多人 / 持久化對話。

## 11. 開發流程(superpowers)

本 v2 一樣走 superpowers 流水線:brainstorming(本文件)→ writing-plans → subagent-driven TDD + requesting-code-review → finishing-a-development-branch。**註**:`ingest/prompts_zh/rag_eval/rag_app` 已先以 spike 形式寫出(偏離流程),計畫會把它們納入並**一律送 code review**。
