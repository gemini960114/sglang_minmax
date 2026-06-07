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

```bash
singularity pull /work/$USER/docker/sglang_latest.sif docker://lmsysorg/sglang:latest
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

SGLang 啟動 MiniMax-M2.7 共經歷以下階段：

| 階段 | 說明 | 預估時間 |
|------|------|---------|
| 1. 模型載入 | 讀取 125 個 safetensors（共 215 GB）| ~1~2 min |
| 2. Gloo 初始化 | 4 個 TP rank 建立通訊 | ~30s |
| 3. DeepGEMM warmup | 首次啟動預編譯 16,384 種 CUDA kernel（H200 專用 cache）| ~5~10 min |
| 4. MoE Triton 配置 | FP8 MoE kernel 初始化 | ~30s |
| **5. 就緒** | `The server is fired up and ready to roll!` | — |

> **DeepGEMM cache**：首次啟動後 kernel 會快取於 `~/.cache/deepgemm/`，後續重啟此階段大幅縮短。

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
