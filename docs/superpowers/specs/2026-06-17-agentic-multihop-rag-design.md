# HopRAG — Agentic Multi-Hop RAG 設計文件

- **日期**：2026-06-17
- **狀態**：設計已通過，待寫實作計畫
- **類型**：個人 side project（技能深化 + 求職作品集）

---

## 1. 目標與成功標準

做一個**你自己擁有編排迴圈**的 agentic 檢索器，在標準多跳 QA benchmark 上、用一個**固定的本地 dense retriever**，證明它比 naive RAG 更準，並用 ablation 把每個 agentic 機制的貢獻拆開展示。

作品集要能講出三句可量化的話：

1. 同一個 retriever、同一個生成模型，agentic 迴圈在 HotpotQA 上的答案 EM/F1 比 naive RAG 高 **X%**。
2. Ablation：拿掉 query 分解掉 _%，拿掉充分性自檢掉 _%，拿掉引用驗證使 supporting-fact recall 掉 _%。
3. Accuracy vs Claude 呼叫次數的成本曲線，證明提升值得那些額外呼叫。

第三點是把本專案跟「只貼一個 chat UI」的作品區隔開的關鍵——重點是 agent **工程**與**可量測的取捨**，不是 demo 畫面。

**非目標（明確排除，避免範圍蔓延）**：對話式前端 UI、自建語料、超越 SOTA、線上服務化、多語言。

---

## 2. 核心架構決策（brainstorming 拍板）

| 面向 | 決策 | 理由 |
|---|---|---|
| 方向 | Agentic 多跳 RAG | agentic 策略是看得見、可 ablate 的 IP |
| LLM | Claude CLI headless（`claude -p`，JSON 輸出）當**每步模型** | 省掉 API/重試/串流 plumbing；dogfood Claude 生態 |
| 編排 | **使用者擁有迴圈**，不把多跳決策交給 Claude Code 通用 loop | agentic 邏輯必須是自己寫的、講得清楚的 |
| Retriever | 本地 dense embedding，**全變體共用同一個** | 提升乾淨歸因到 agentic 迴圈，而非更猛的 retriever |
| Benchmark | HotpotQA + **池化語料** | 現成標準答案 + supporting facts → 全自動算分；池化＝真正的檢索問題 |
| Stack | Python + LanceDB（預設，可換） | embedding / `datasets` / 指標生態最順 |

註：Claude CLI 只提供生成/推理，**不提供 embedding**，故 retriever 這層用本地模型，與「Claude CLI 只負責 LLM」相容。

---

## 3. 元件設計

每個元件單一職責、介面明確、可獨立測試。

| 元件 | 職責 | 介面（概念） | 依賴 |
|---|---|---|---|
| `Indexer` | 把語料切成 chunk → 本地 embed → 建索引 | `build(paragraphs) -> index` | sentence-transformers、LanceDB |
| `Retriever` | 固定的 dense 檢索；naive/agentic/ablation 共用 | `search(query, k) -> list[Chunk]` | index |
| `ClaudeClient` | 包 `claude -p --output-format json` 的 subprocess；支援結構化輸出 + schema 驗證 | `complete(prompt) -> text`；`complete_json(prompt, schema) -> dict` | Claude CLI on PATH |
| `NaiveRAG` | Baseline：單次檢索 → 單次作答 | `answer(question) -> Result` | Retriever, ClaudeClient |
| `AgenticRAG` | **主角**：第 4 節的五段迴圈；各機制可開關（ablation 用） | `answer(question) -> Result`（含 `trace`） | Retriever, ClaudeClient |
| `EvalHarness` | 載入 benchmark 子集 → 跑某 pipeline → 算指標 → 出報告 | `run(pipeline, questions) -> Report` | datasets loader、metrics |
| `Report` | 作品集產物：結果表 + 成本曲線 + 數條多跳推理 trace | — | — |

`Result` 結構：`{answer, cited_chunks, trace, n_claude_calls, n_retrievals}`，`trace` 記錄每一跳的 query / 檢索到的 chunk / Claude 的判斷，供除錯與 demo。

---

## 4. Agentic 迴圈（IP，五個可 ablate 的機制）

```
question
  → ① 分解 / 規劃 (Claude)         產生初始子問題與檢索策略
  → loop (硬上限 max_hops，預設 5):
        retrieve(current_query)      ← 固定 retriever，從池化語料檢索
        → ② 多跳推理 (Claude)        讀檢索結果，用「剛學到的事實」組下一跳 query
        → ③ 充分性自檢 (Claude)      「證據夠回答了嗎？」→ 決定是否跳出迴圈
  → ④ 合成 + 帶引用 (Claude)        產生答案，引用具體 chunk id
  → ⑤ 引用驗證 (Claude)             逐項檢查每個 claim 是否被所引 chunk 支持，不支持就修正/移除
  → Result(answer, cited_chunks, trace, costs)
```

機制 ② 是擊敗 naive 的核心：naive RAG 只用原始問題檢索一次，永遠拿不到那個「要先答出第一跳才找得到」的橋接段落；agentic 迴圈用第一跳學到的實體去組第二跳的 query。

**機制開關（for ablation）**：`AgenticRAG` 以 config 控制 ①②③⑤ 是否啟用；關掉 ③ 時改用固定跳數。

---

## 5. 資料與 Eval 設計

### 5.1 語料與池化
- 取 HotpotQA dev 的一個子集（headline 用約 300–500 題；開發期用 20–30 題的快速集）。
- 每題附 10 段（2 段 gold + 8 干擾），各段帶 Wikipedia 標題。
- **池化**：把子集內所有段落收集、依（標題+內文）去重，建成**一個共享索引**。所有變體都從這個池檢索（而非只看自己那 10 段）→ 這是真正的檢索問題，但語料有界、離線。
- **Chunk 粒度**：一段＝一個 chunk（HotpotQA 段落本就段落大小），metadata 帶標題與來源題目。

### 5.2 對照組
- **B0 NaiveRAG**：對原始問題檢索 top-k → 塞進單次 Claude 作答。
- **B1 AgenticRAG（full）**：①②③④⑤ 全開。
- **Ablations**：B1 −①（不分解）／ B1 −③（改固定跳數）／ B1 −⑤（不驗證引用）。
- 所有變體共用同一 retriever + 同一 embedder + 同一 Claude 模型。

### 5.3 指標
- **答案品質**：EM、F1（對 gold answer，HotpotQA 官方算法）。
- **證據品質**：supporting-fact recall / precision（段落層級：是否檢索到 gold 段落）。
- **成本**：每題平均 Claude 呼叫數、平均檢索次數（有 token 數則一併記）。
- **報告**：表格（各變體 × 各指標）+ accuracy/cost 曲線 + 3–5 條質化 trace。

### 5.4 成本控管
agentic 迴圈每題約 5–8 次 Claude 呼叫；500 題 × ~6 ≈ 3000 次呼叫，非零成本。開發期一律用小快速集，full eval 只在出報告時跑一次。eval 結果落地存檔，避免重跑。

---

## 6. Stack 與預設選擇（可調）

- **語言**：Python
- **Embedding**：`BAAI/bge-small-en-v1.5`（英文 benchmark；小、快、離線）
- **向量庫**：LanceDB（本地、零服務）
- **LLM**：`claude -p`；結構化步驟（①③⑤）用 `--output-format json` + schema 驗證
- **資料集載入**：HuggingFace `datasets`

---

## 7. 錯誤處理 / 韌性

- subprocess 逾時 + 指數退避重試。
- Claude JSON 輸出壞掉 → 帶 schema 重新 prompt（有限次數），仍失敗則記錄並降級。
- 迴圈失控防護：`max_hops` 硬上限。
- 空檢索 / 無結果的 fallback 路徑。
- 所有外部呼叫的錯誤都寫入 `trace`，便於事後分析。

---

## 8. 測試策略（TDD）

- **單元**：`Retriever`（已知 query 命中已知 chunk）、`ClaudeClient` JSON 解析（mock subprocess，不真打 CLI）、指標函式（已知 (pred, gold) 算 EM/F1）。
- **整合**：2–3 題 fixture 的端到端 smoke test（naive 與 agentic 各一），用 mock 或極小真實呼叫。
- **eval 可重現**：固定隨機種子與子集，報告數字可重跑復現。

---

## 9. v1 範圍（YAGNI）

- **做**：§3 全部元件 + HotpotQA 池化語料 + B0/B1 + 3 個 ablation + §5.3 指標 + §5.3 報告。
- **之後（明確 out）**：HotpotQA fullwiki / MuSiQue、hybrid 檢索、reranking、自建 demo 語料、前端 UI、服務化。

---

## 10. 開發流程（使用 superpowers skills）

本專案的**開發過程**走完整條 superpowers 流水線（superpowers 是工作方式，不是專案題材）：

1. `brainstorming`（本文件產出，已完成）
2. `writing-plans` — 把本 spec 變成逐步實作計畫
3. 實作 — `test-driven-development` + `subagent-driven-development`（或 `executing-plans`）
4. 收尾 — `requesting-code-review` + `verification-before-completion` + `finishing-a-development-branch`
