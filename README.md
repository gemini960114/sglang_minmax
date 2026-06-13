# HPC MiniMax-M2.7 Serving with SGLang

此專案專注於在高效能運算 (HPC) 環境中，利用 **SGLang** 在 **NVIDIA H200 × 4 (TP=4)** 叢集上部署與執行 **MiniMax-M2.7** 模型的完整解決方案。

## 🚀 核心特點

- **MiniMax-M2.7 深度優化**：針對 MiniMax 特有的 MLA 架構與 Tool Call / Reasoning 功能，配置專用解析器（`minimax-m2` & `minimax-append-think`）。
- **動態 GPU 偵測**：啟動腳本自動偵測可用 GPU 數量並設定最優 `--tp-size` / `--ep-size`。
- **高效能實測**：H200 × 4 環境下，**500 人並發 0 失敗，系統吞吐量達 13,757 tok/s**。
- **HPC 環境適配**：自動處理 CUDA 路徑、Triton 編譯器衝突（強制使用 GCC 12.2）。

---

## 🛠 方案一：原生環境安裝 (Native / uv)

推薦使用 `uv` 管理環境，確保 Python 表頭檔與依賴完整。

### 1. 安裝 uv（若尚未安裝）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. 執行安裝腳本

此腳本自動建立 `.venv`（Python 3.11）並安裝 SGLang：

```bash
bash sglang_install.sh
```

### 2.5 修補 Claude Code 相容性 (選用)

如果您需要直接將 **Claude Code CLI** 對接至 SGLang 服務，請在安裝後執行此腳本來修補您虛擬環境中的套件：

```bash
source .venv/bin/activate
python patch_claude_compatibility.py
```

### 3. 啟動服務

```bash
bash launch_minimax.sh
```

- **Port**：`8000`
- **硬體需求**：建議 4 張 H200 / H100（Tensor Parallel = 4）
- **模型**：`MiniMaxAI/MiniMax-M2.7`（215 GB，125 個 safetensors shards）

### 4. 使用 Slurm 背景提交

```bash
sbatch launch_minimax.slurm
```

---

## 🐋 方案二：容器化部署 (Singularity)

若主機 OS 的 `glibc` 版本過舊或環境隔離需求高，建議使用 Singularity 鏡像。

### 1. 下載鏡像

#### MiniMax-M2.7 (最新穩定版)
```bash
singularity pull /work/$USER/docker/sglang_latest.sif docker://lmsysorg/sglang:latest
```

#### MiniMax-M3 版本
```bash
mkdir -p /work/$(whoami)/containers/
singularity pull --force /work/$(whoami)/containers/sglang-dev.sif docker://lmsysorg/sglang:dev-cu13-minimax-m3
```

### 1.5 下載模型權重 (建議於登入節點執行)

由於計算節點通常無法連外網，請在登入節點先下載模型權重至快取目錄：

```bash
export HF_HOME="/work/$(whoami)/huggingface_cache"

# 下載 MiniMax-M2.7
hf download MiniMaxAI/MiniMax-M2.7

# 下載 MiniMax-M3 (BF16 原版)
hf download MiniMaxAI/MiniMax-M3
```


### 2. 啟動服務

```bash
bash launch_minimax_sglang_singularity.sh
```

### 3. 使用 Slurm 背景提交（推薦長時間運行）

```bash
# 原生環境版
sbatch launch_minimax.slurm

# 容器環境版
sbatch launch_minimax_singularity.slurm
```

---

## 🧪 測試驗證

### 啟動後確認伺服器就緒

```bash
# 健康檢查（回傳 {} 即代表就緒）
curl http://127.0.0.1:8000/health

# 或用 Python 腳本（自動等待伺服器啟動）
source .venv/bin/activate
python test_inference.py
```

> **注意**：MiniMax-M2.7 啟動時包含 **DeepGEMM kernel warmup**（首次啟動需額外約 5~10 分鐘），完成後 API 才可用。

### 推論測試 (`test_inference.py`)

```bash
source .venv/bin/activate

# 在計算節點上執行（localhost 自動有效）
python test_inference.py

# 從登入節點指定計算節點名稱
python test_inference.py --host 25a-hgpn004

# 用環境變數（設一次，之後都有效）
export SGLANG_HOST=25a-hgpn004
python test_inference.py

# 指定 port 與模型
python test_inference.py --host 25a-hgpn004 --port 8000 --model MiniMaxAI/MiniMax-M2.7

# 跳過等待直接測試
python test_inference.py --no-wait
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--host` | `localhost`（可用 `SGLANG_HOST` 環境變數覆蓋）| 伺服器主機名或 IP |
| `--port` | `8000` | 服務 port |
| `--model` | `MiniMaxAI/MiniMax-M2.7` | 模型名稱 |
| `--no-wait` | — | 跳過健康等待，直接測試 |

---

## 📊 效能實測報告 (Performance Benchmark)

### 環境

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA H200 × 4 |
| Tensor Parallel | 4 |
| 模型 | MiniMaxAI/MiniMax-M2.7 |
| max_tokens | 512 |
| SGLang | Continuous Batching |

### 壓力測試結果

| 人數 | 成功 | 失敗 | 吞吐量 (tok/s) | p50 延遲 | p95 延遲 | p99 延遲 |
|------|------|------|----------------|---------|---------|---------|
| 10 | 10 | 0 | 851 | 6.02s | 6.02s | 6.02s |
| 50 | 50 | 0 | 2,843 | 9.00s | 9.00s | 9.00s |
| 100 | 100 | 0 | 4,825 | 10.61s | 10.61s | 10.61s |
| 200 | 200 | 0 | 7,137 | 14.34s | 14.34s | 14.34s |
| 300 | 300 | 0 | 10,168 | 15.09s | 15.10s | 15.10s |
| 400 | 400 | 0 | 12,150 | 16.83s | 16.84s | 16.85s |
| **500** | **500** | **0** | **13,255** | **19.26s** | **19.30s** | **19.30s** |

### MiniMax-M3 壓力測試結果 (H200 × 8, TP=8, EP=8)

```text
============================================================
📈 階梯測試總覽
────────────────────────────────────────────────────────────
    人數     成功     失敗    吞吐(tok/s)   p50(s)   p95(s)   p99(s)
────────────────────────────────────────────────────────────
   100    100      0        3,028    16.76    16.77    16.77
   200    200      0        4,827    21.00    21.03    21.03
   300    300      0        6,130    24.73    24.77    24.78
   400    400      0        7,185    27.78    28.07    28.07
   500    500      0        8,236    30.71    30.77    30.78
============================================================
```

### 分析摘要

- **承載上限**：**400~500 人並發**（延遲 < 20s），0 失敗率
- **推薦並發數**：**200~300 人**（吞吐量成長最快、延遲仍合理）
- **吞吐量峰值**：**13,000~15,000 tok/s**（預估）
- **拐點**：400→500 人時，邊際增益從 +3,031 tok/s 降至 +1,105 tok/s，為系統飽和訊號

### 壓力測試 (`test_concurrency.py`)

```bash
source .venv/bin/activate

# 在計算節點上執行（localhost 自動有效）
python test_concurrency.py --users 100

# 從登入節點指定計算節點
python test_concurrency.py --host 25a-hgpn004 --users 100

# 用環境變數
export SGLANG_HOST=25a-hgpn004
python test_concurrency.py --ramp

# 自定義階梯步驟
python test_concurrency.py --ramp-steps 100 200 300 400 500

# 自動產生到最大值（每 500 一梯）
python test_concurrency.py --ramp-max 2000

# 調整 max_tokens 與 timeout
python test_concurrency.py --users 200 --max-tokens 1024 --timeout 600
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--host` | `localhost`（可用 `SGLANG_HOST` 環境變數覆蓋）| 伺服器主機名或 IP |
| `--port` | `8000` | 服務 port |
| `--users` | `10` | 同時模擬使用者數 |
| `--ramp` | — | 階梯測試：10→50→100→200→500→1000→2000 |
| `--ramp-steps` | — | 自定義階梯，例如 `--ramp-steps 100 300 500` |
| `--ramp-max` | — | 自動產生到最大人數 |
| `--max-tokens` | `512` | 每次請求最大 token 數 |
| `--timeout` | `300` | 單次請求 timeout 秒數 |

---

## ⚙️ 啟動流程說明

SGLang 啟動模型共經歷以下階段：

### 1. MiniMax-M2.7 (TP=4) 啟動流程

| 階段 | 說明 | 預估時間 |
|------|------|---------|
| 1. 模型載入 | 讀取 125 個 safetensors（共 215 GB）| ~1~2 min |
| 2. Gloo 初始化 | 4 個 TP rank 建立通訊 | ~30s |
| 3. DeepGEMM warmup | 首次啟動預編譯 16,384 種 CUDA kernel（H200 專用 cache）| ~5~10 min |
| 4. MoE Triton 配置 | FP8 MoE kernel 初始化 | ~30s |
| **5. 就緒** | `The server is fired up and ready to roll!` | — |

> **DeepGEMM cache**：首次啟動後 kernel 會快取於 `~/.cache/deepgemm/`，後續重啟此階段大幅縮短。

### 2. MiniMax-M3 (TP=8, EP=8) 啟動流程

| 階段 | 說明 | 預估時間 |
|------|------|---------|
| 1. 模型載入 | 讀取 59 個 safetensors 權重分片 | ~6~7 min (約 397s) |
| 2. NCCL/分散式通訊 | 8 個 TP/EP rank 建立通訊與初始化 | ~15s |
| 3. KV Cache 分配 | 載入並分配推論所需要的 GPU 記憶體 | ~10s |
| 4. CUDA Graph 擷取 | 擷取並優化 52 個 Batch Size 級距的運作圖 (Decode Capture) | ~4~5 min (約 271s) |
| **5. 就緒** | 開始提供 API 服務，支援多模態與推論 | — |

---

## 🤖 Claude Code 整合相容性

本專案已適配 **Claude Code CLI (v2.1.154+)** 的直接連線。

由於新版 Claude Code 在 `/v1/messages` 請求中帶有非標準角色（如 `ctx`, `system`, `msg`），會導致原生的 SGLang 拋出 `400 Pydantic Validation Error`。我們已對虛擬環境中的 SGLang 進行了熱修復，使其能夠順暢對接。

### 1. 本地虛擬環境修補

若為新安裝環境，請在啟用虛擬環境後執行此腳本來修補 SGLang 套件：

```bash
source .venv/bin/activate
python patch_claude_compatibility.py
```

### 1.5 Singularity 容器環境修補 (Bind Mount)

由於 Singularity 鏡像（SIF 檔案）為唯讀狀態，無法直接在容器內執行修補腳本。本專案透過 `Bind Mount` 機制實現容器內修補：

1. **生成修補檔**：當您在主機端啟用虛擬環境並執行 `python patch_claude_compatibility.py` 後，腳本除了修補本機套件，還會自動將修補後的 `protocol.py` 與 `serving.py` 輸出到 `patched_anthropic/` 目錄中。
2. **自動掛載**：Singularity 啟動腳本（如 `launch_minimax_m3_sglang_singularity.sh`）會自動偵測 `patched_anthropic/` 目錄是否存在，若存在則會自動在啟動容器時進行掛載（Bind Mount），覆蓋容器內部的 Anthropic API 協定程式碼，使其支援 Claude Code。


<details>
<summary><b>🔍 點此查看修補原理與程式碼變更 (Diffs)</b></summary>

#### 檔案 1: 放寬 API 欄位驗證 (`protocol.py`)
在 `.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/anthropic/protocol.py` 中放寬 `AnthropicMessage` 結構中 `role` 欄位的限制，納入 `"system"`, `"ctx"`, `"msg"`：
```diff
class AnthropicMessage(BaseModel):
    """Message structure"""

-   role: Literal["user", "assistant"]
+   role: Literal["user", "assistant", "system", "ctx", "msg"]
    content: str | list[AnthropicContentBlock]
```

#### 檔案 2: 適配轉換至 OpenAI 格式 (`serving.py`)
在 `.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/anthropic/serving.py` 中，將 `"system"` 與 `"ctx"` 映射至 OpenAI 的 `"system"`，並將其餘自訂角色映射至 `"user"`：
```diff
        # Convert messages
        for msg in anthropic_request.messages:
+             role = msg.role
+             if role == "ctx":
+                 role = "system"
+             elif role not in ["user", "assistant", "system"]:
+                 role = "user"
+ 
              if isinstance(msg.content, str):
-                 openai_messages.append({"role": msg.role, "content": msg.content})
+                 openai_messages.append({"role": role, "content": msg.content})
                  continue

              # Complex content with blocks
-             openai_msg = {"role": msg.role}
+             openai_msg = {"role": role}
              content_parts = []
              ...
                      # Tool results from user become separate tool messages
-                     if msg.role == "user":
+                     if role == "user":
                          openai_messages.append(
                              {
                                  "role": "tool",
                                  "tool_call_id": tool_call_id,
                                  "content": tool_content,
                              }
                          )
```
</details>

### 2. Claude Code 設定步驟與設定檔範本 (`~/.claude/settings.json`)

請依照以下步驟設定您的 Claude Code：

1. **建立/開啟設定檔**：在您本地的終端機執行編輯指令，開啟設定檔（若目錄或檔案不存在會自動建立）：
   ```bash
   mkdir -p ~/.claude
   nano ~/.claude/settings.json
   ```
2. **複製貼入設定內容**：將下方提供的 JSON 範本完整複製並貼入檔案中。
3. **動態更新計算節點**：
   * 透過 `squeue -u $USER` 確認您目前正在運作的 Slurm Job 位於哪一個計算節點（例如：`25a-hgpn144`）。
   * 將 JSON 中的 `"ANTHROPIC_BASE_URL"` 的 `<當前Slurm計算節點名稱>` 替換為該計算節點（例如將其改為 `"http://25a-hgpn144:8000"`）。
4. **啟動 Claude Code**：存檔離開後，在終端機輸入 `claude` 啟動，即可開始與 SGLang 服務進行連線對話。

#### 設定檔範本內容：

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "sk-local",
    "ANTHROPIC_BASE_URL": "http://<當前Slurm計算節點名稱>:8000",
    "ANTHROPIC_MODEL": "MiniMaxAI/MiniMax-M2.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMaxAI/MiniMax-M2.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME": "MiniMaxAI/MiniMax-M2.7",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMaxAI/MiniMax-M2.7",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME": "MiniMaxAI/MiniMax-M2.7",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMaxAI/MiniMax-M2.7",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME": "MiniMaxAI/MiniMax-M2.7",
    "CLAUDE_CODE_DISABLE_THINKING": "1",
    "LITELLM_DROP_PARAMS": "true",
    "ANTHROPIC_DISABLE_THINKING": "1",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  },
  "attribution": {
    "commit": "",
    "pr": ""
  },
  "model": "opus",
  "effortLevel": "max",
  "promptSuggestionEnabled": false,
  "plansDirectory": "./plans",
  "prefersReducedMotion": true,
  "theme": "dark",
  "terminalProgressBarEnabled": false
}
```

> [!NOTE]
> * `ANTHROPIC_BASE_URL` 後方請**不要**加上 `/v1`。
> * 如果 Slurm 工作重啟，請使用 `squeue -u c00cjz00` 確認新的計算節點，並更新 `ANTHROPIC_BASE_URL` 中的節點名稱。

---

## 📝 檔案說明

| 檔案 | 說明 |
|------|------|
| `sglang_install.sh` | 使用 `uv` 建立 Python 3.11 虛擬環境並安裝 SGLang |
| `launch_minimax.sh` | 原生環境啟動腳本（自動偵測 GPU 數量） |
| `launch_minimax_sglang_singularity.sh` | Singularity 容器啟動腳本 |
| `launch_minimax.slurm` | 原生環境 Slurm 任務腳本（gres=gpu:H200:4） |
| `launch_minimax_singularity.slurm` | 容器環境 Slurm 任務腳本 |
| `test_inference.py` | 推論測試工具，支援健康檢查等待、CLI 參數 |
| `test_concurrency.py` | 多用戶并發壓力測試，支援階梯模式與 p95/p99 統計 |

---

*Last updated: 2026-06-07 ・ Tested on NVIDIA H200 × 4 ・ MiniMaxAI/MiniMax-M2.7*
