# HPC MiniMax-M2.5 Serving with SGLang

此專案專注於在高效能運算 (HPC) 環境（如 NVIDIA H100 叢集）中，利用 **SGLang** 部署與執行 **MiniMax-M2.5** 模型的完整解決方案。

## 🚀 核心特點
- **MiniMax-M2.5 深度優化**：針對 MiniMax 特有的 MLA 架構與 Tool Call/Reasoning 功能，配置了專用的解析器 (`minimax-m2` & `minimax-append-think`)。
- **高效能產出**：在 H100 (TP=4) 環境下，單併發生成速度可達 **125 tokens/s** 以上。

- **HPC 環境適配**：自動處理 `glibc` 相容性、CUDA 路徑對齊以及 Triton 編譯器衝突（強制使用 GCC 12.5.0）。

---

## 🛠 方案一：原生環境安裝 (Native Installation)

推薦使用 `uv` 進行環境管理，以確保 Python 表頭檔與依賴項的完整性。

### 1. 執行安裝腳本
此腳本會自動建立 `.venv` 並安裝 SGLang：
```bash
bash sglang_install.sh
```

### 2. 啟動服務
使用優化過的啟動腳本，自動配置 CUDA 12.5 與 GCC 環境：
```bash
bash launch_minimax.sh
```
*   **埠號**: 8000
    *   **硬體需求**: 建議 4 張 GPU (Tensor Parallel = 4)

---

## 🐋 方案二：容器化部署 (Singularity) - 推薦

若主機 OS 的 `glibc` 版本過舊，建議使用 Singularity 鏡像以獲得最穩定的執行環境。

### 1. 下載鏡像
```bash
singularity pull /work/$USER/docker/sglang_latest.sif docker://lmsysorg/sglang:latest
```

### 2. 啟動服務
此腳本封裝了 Singularity 指令與 MiniMax 專用參數：
```bash
bash launch_minimax_sglang_singularity.sh
```

### 3. 使用 Slurm 提交任務 (推薦背景執行)
若需要在叢集節點背景執行服務，請使用 Slurm 腳本：
- **標準版**: `sbatch launch_minimax.slurm`
- **高併發/多人優化版**: `sbatch launch_minimax_multi.slurm`

*   **特性**: 自動分配 4 張 GPU，解決主機環境庫版本衝突，適合長時間運行服務。多人優化版額外啟用了 `LPM` 排程與 `Chunked Prefill` 以降低集體等待時間。


---

## 🧪 測試驗證

服務啟動後（預設端口 `8000`），使用 Python 腳本驗證 OpenAI 相容 API 的回應：

```bash
source .venv/bin/activate
python test_inference.py
```

若成功連線，您將看到模型輸出的生成內容與速度統計。

---

## 📊 效能實測報告 (Performance Benchmark)

在 NVIDIA H100 (80GB) x 4 (TP=4) 的環境下進行壓力測試，結果如下：

- **吞吐量天花板**：系統總吞吐量穩定維持在 **4,500 - 5,000 tokens/s** 之間。
- **使用者承載力**：
    - **500 人同時併發**：每人平均延遲約 **33 秒**（可獲得 512 tokens 回應）。此狀態下系統運行極其強悍且健康。對於非即時回覆場景（如批次公文、報告生成、程式碼分析）完全可以接受。
    - **最佳體驗區**：若追求極致流暢體感（每人 10 秒內拿到結果），建議將併發人數控制在 **150 - 200 人**。
- **顯存利用率**：在 500 人併發下，Token 緩存 (KV Cache) 利用率僅約 **5%**，顯示系統在處理超長文本或更高併發上仍有巨大潛力。

---

## 📝 重要檔案說明
- `sglang_install.sh`: 使用 `uv` 建立 Python 3.11 環境並安裝 SGLang。
- `launch_minimax.sh`: 原生環境下的 MiniMax-M2.5 啟動腳本。
- `launch_minimax_sglang_singularity.sh`: 容器化環境下的 MiniMax-M2.5 啟動腳本（解決相容性問題）。
- `test_inference.py`: 快速測試工具，支援串流輸出與 Token 速度計算。

---
*Created by Antigravity AI Assistant.*
