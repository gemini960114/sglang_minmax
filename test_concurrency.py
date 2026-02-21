import asyncio
import aiohttp
import time
import statistics

# ==============================================================================
# SGLang 多併發測試壓力測試工具 (Concurrent Benchmark)
# ==============================================================================

# 配置參數
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "MiniMaxAI/MiniMax-M2.5"
CONCURRENT_USERS = 10  # 同時模擬多少人
REQUESTS_PER_USER = 1  # 每個人發送幾次請求
PROMPT = "請用繁體中文寫一篇關於人工智慧未來發展的五百字文章。"

async def send_request(session, user_id):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }
    
    start_time = time.perf_counter()
    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                end_time = time.perf_counter()
                latency = end_time - start_time
                tokens = result['usage']['completion_tokens']
                tps = tokens / latency
                print(f"[User {user_id}] 成功: {tokens} tokens, 耗時: {latency:.2f}s, 速度: {tps:.2f} tps")
                return latency, tokens
            else:
                print(f"[User {user_id}] 失敗: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"[User {user_id}] 錯誤: {e}")
        return None

async def run_benchmark():
    print(f"🚀 開始壓力測試...")
    print(f"模擬人數: {CONCURRENT_USERS}")
    print(f"目標模型: {MODEL_ID}")
    print("-" * 40)

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(CONCURRENT_USERS)]
        
        overall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        overall_end = time.perf_counter()

    # 數據統計
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("❌ 沒有成功的請求。")
        return

    latencies = [r[0] for r in valid_results]
    total_tokens = sum(r[1] for r in valid_results)
    total_time = overall_end - overall_start
    
    print("-" * 40)
    print(f"📊 測試結果總結:")
    print(f"總成功次數: {len(valid_results)}/{CONCURRENT_USERS}")
    print(f"總耗時: {total_time:.2f} 秒")
    print(f"平均每人延遲: {statistics.mean(latencies):.2f} 秒")
    print(f"系統總吞吐量: {total_tokens / total_time:.2f} tokens/s (Throughput)")
    print(f"單一請求中位數延遲: {statistics.median(latencies):.2f} 秒")
    print("-" * 40)

if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        pass
