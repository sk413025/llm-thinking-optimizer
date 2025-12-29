# Ollama Feedback Descent 實驗

使用 Feedback Descent (FD) 優化 prompt，讓 LLM 在 tool call 前產生推理過程。

## 實驗目的

**目標**: 不經過 fine-tuning，僅通過 prompt 優化使模型學會「先思考、再行動」的模式。

**方法**: Feedback Descent - 使用較大的評估器模型對輸出進行 pairwise comparison，迭代改進 prompt。

**結果**:
- Thinking Rate: 26.7% → **100%**
- Tool Accuracy: 66.7% → **60%** (維持合理水平)

---

## 系統需求

| 項目 | 需求 |
|------|------|
| OS | Linux (Ubuntu 22.04+) |
| GPU VRAM | 16 GB+ (for gpt-oss:20b evaluator) |
| RAM | 16 GB+ |
| Python | 3.10+ |
| 儲存空間 | 30 GB (for models) |

---

## 快速開始

### 1. 安裝 Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. 啟動 Ollama 服務

```bash
ollama serve
```

### 3. 下載模型 (新終端)

```bash
# 必要模型
ollama pull mistral:7b      # 目標模型 (7.2B, ~5GB VRAM)
ollama pull gpt-oss:20b     # 評估器 (20.9B, ~12GB VRAM)

# 可選 - 基線測試用
ollama pull llama3.1:8b
ollama pull qwen3:8b
```

### 4. 安裝 Python 依賴

```bash
pip install requests
```

### 5. 運行實驗

```bash
# (可選) 運行基線測試 - 確認模型支持 thinking
python ollama_baseline_test.py

# 運行 FD 優化
python ollama_fd_optimizer.py
```

---

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `ollama_fd_optimizer.py` | FD 優化器主程式 |
| `ollama_baseline_test.py` | 基線測試腳本 |
| `FD_DEBUGGING_NOTES.md` | 調試記錄與已知問題 |
| `reference_results/` | 參考結果 (用於驗證) |

---

## 配置說明

編輯 `ollama_fd_optimizer.py` 頂部的配置：

```python
# 模型配置
TARGET_MODEL = "mistral:7b"        # 被優化的模型
EVALUATOR_MODEL = "gpt-oss:20b"    # 評估器模型
OLLAMA_BASE_URL = "http://localhost:11434"

# FD 超參數
MAX_ITERATIONS = 15      # 最大優化輪數
EARLY_STOP_K = 5         # 連續 K 輪無改進則停止
SAMPLES_PER_PROMPT = 3   # 每個 prompt 採樣次數
TEMPERATURE = 0.7        # 生成溫度
MIN_TOOL_ACCURACY = 0.4  # 最低 tool accuracy 門檻
```

---

## 預期結果

### FD 優化輸出

```
[Iteration 0] Initial evaluation...
  Thinking Rate: 26.7%
  Tool Accuracy: 66.7%
  Score: 0.427

[Iteration 1] Generating improved prompt...
  Thinking Rate: 100.0%
  Tool Accuracy: 60.0%
  Score: 0.840
  ✓ Accepting new prompt (improvement)

[Early Stop] Target thinking rate achieved!
```

### 輸出檔案

```
ollama_fd_results/
├── best_prompt_[timestamp].json      # 最佳 prompt
├── optimization_log_[timestamp].json  # 迭代記錄
└── summary_[timestamp].json           # 結果摘要
```

### 結果驗證

```bash
# 查看結果摘要
cat ollama_fd_results/summary_*.json

# 預期數值:
# - final_thinking_rate: 1.0 (100%)
# - final_tool_accuracy: ~0.6 (60%)
```

---

## 故障排除

### Tool Accuracy 為 0%

**原因**: 解析器可能無法識別某些格式的 tool call。

**解決**: 參見 `FD_DEBUGGING_NOTES.md` 中的多格式解析器說明。

### Ollama 連接失敗

```bash
# 確認 Ollama 服務正在運行
curl http://localhost:11434/api/tags

# 如果失敗，重新啟動
ollama serve
```

### GPU VRAM 不足

如果 GPU VRAM < 16GB，可以嘗試：

1. 使用較小的評估器:
```python
EVALUATOR_MODEL = "mistral:7b"  # 改用相同大小的模型
```

2. 或使用 llama3.1:8b 作為目標:
```python
TARGET_MODEL = "llama3.1:8b"
EVALUATOR_MODEL = "mistral:7b"
```

---

## 替換模型

### 使用其他目標模型

需求：
- 支持 Ollama tool calling API
- 參數量 7B+（需要足夠的 ICL 能力）

```python
TARGET_MODEL = "your-model:tag"
```

### 使用其他評估器

建議：
- 評估器參數量 >= 目標模型
- 不同架構可提供多元視角

```python
EVALUATOR_MODEL = "your-evaluator:tag"
```

---

## 參考結果

`reference_results/` 目錄包含原始實驗結果，可用於比對驗證：

| 檔案 | 說明 |
|------|------|
| `ollama_baseline_summary.json` | 4 個模型的基線測試結果 |
| `ollama_fd_results/best_prompt_*.json` | 優化後的最佳 prompt |
| `ollama_fd_results/summary_*.json` | 優化結果摘要 |

---

## 實驗背景

### 為什麼需要 FD？

直接 fine-tuning 小模型添加 thinking 能力成本高。FD 嘗試通過 prompt 工程達到類似效果。

### 為什麼選擇 7B+ 模型？

之前對 FunctionGemma-270M 的實驗顯示，小模型缺乏 in-context learning 能力：
- Thinking Rate = 0% (所有 prompt 變體)
- Output Diversity = 0.00 (輸出完全固定)

7B+ 模型具備足夠的 ICL 能力響應 prompt 變化。

### 實驗結論

- FD 在 7B+ 模型上可行
- mistral:7b 達到 100% thinking rate
- Tool accuracy 維持 60%（可接受的 trade-off）

---

## 引用

如果這個實驗對你有幫助，歡迎引用：

```
Feedback Descent for LLM Thinking Capability
- 使用 pairwise comparison 優化 prompt
- 目標：讓模型「先思考、再行動」
- 結果：thinking rate 26.7% → 100%
```

---

## License

MIT License
