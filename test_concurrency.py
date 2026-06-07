import asyncio
import aiohttp
import time
import statistics
import argparse

# ==============================================================================
# SGLang 多併發壓力測試工具 (Concurrent Benchmark)
# 用法:
#   python test_concurrency.py --users 50
#   python test_concurrency.py --users 100 --max-tokens 1024
#   python test_concurrency.py --ramp              # 自動階梯: 10→50→100→200→500
# ==============================================================================

API_URL   = "http://localhost:8000/v1/chat/completions"
MODEL_ID  = "MiniMaxAI/MiniMax-M2.7"
PROMPT    = "請用繁體中文寫一篇關於人工智慧未來發展的五百字文章。"

async def send_request(session, user_id, max_tokens, timeout):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            API_URL, json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                elapsed = time.perf_counter() - start
                tokens  = result["usage"]["completion_tokens"]
                tps     = tokens / elapsed
                print(f"  [User {user_id:>4}] ✅ {tokens:>4} tok  {elapsed:>6.2f}s  {tps:>7.2f} tps")
                return {"latency": elapsed, "tokens": tokens, "ok": True}
            else:
                body = await resp.text()
                print(f"  [User {user_id:>4}] ❌ HTTP {resp.status}: {body[:80]}")
                return {"ok": False}
    except asyncio.TimeoutError:
        print(f"  [User {user_id:>4}] ⏰ Timeout (>{timeout}s)")
        return {"ok": False}
    except Exception as e:
        print(f"  [User {user_id:>4}] 💥 Error: {e}")
        return {"ok": False}

def pct(lst, p):
    """百分位數"""
    lst_sorted = sorted(lst)
    idx = max(0, int(len(lst_sorted) * p / 100) - 1)
    return lst_sorted[idx]

async def run_stage(users, max_tokens, timeout):
    print(f"\n{'='*50}")
    print(f"🚀 模擬人數: {users}  max_tokens: {max_tokens}")
    print(f"{'='*50}")

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request(session, i, max_tokens, timeout) for i in range(users)]
        t0      = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    ok      = [r for r in results if r["ok"]]
    fail    = users - len(ok)
    if not ok:
        print(f"\n❌ 全部失敗！({users}/{users} 失敗)")
        return None

    latencies = [r["latency"] for r in ok]
    tokens    = [r["tokens"]  for r in ok]
    total_tok = sum(tokens)

    print(f"\n{'─'*50}")
    print(f"📊 結果 ({users} 人):")
    print(f"  成功/失敗        : {len(ok)}/{fail}")
    print(f"  總耗時           : {elapsed:.2f} s")
    print(f"  系統吞吐量       : {total_tok/elapsed:,.0f} tok/s")
    print(f"  平均延遲         : {statistics.mean(latencies):.2f} s")
    print(f"  中位數延遲 (p50) : {statistics.median(latencies):.2f} s")
    print(f"  p95 延遲         : {pct(latencies, 95):.2f} s")
    print(f"  p99 延遲         : {pct(latencies, 99):.2f} s")
    print(f"  最大延遲         : {max(latencies):.2f} s")
    print(f"{'─'*50}")

    return {
        "users":      users,
        "success":    len(ok),
        "fail":       fail,
        "throughput": total_tok / elapsed,
        "p50":        statistics.median(latencies),
        "p95":        pct(latencies, 95),
        "p99":        pct(latencies, 99),
    }

async def main():
    parser = argparse.ArgumentParser(description="SGLang 併發壓力測試")
    parser.add_argument("--users",      type=int, default=10,
                        help="同時模擬使用者數 (預設 10)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="每次請求最大 token 數 (預設 512)")
    parser.add_argument("--timeout",    type=int, default=300,
                        help="單次請求 timeout 秒數 (預設 300)")
    parser.add_argument("--ramp",       action="store_true",
                        help="階梯測試模式：10→50→100→200→500→1000→2000")
    parser.add_argument("--ramp-steps", type=int, nargs="+", metavar="N",
                        help="自定義階梯步驟，例如: --ramp-steps 100 200 500 1000 2000")
    parser.add_argument("--ramp-max",   type=int, default=None,
                        help="自動產生階梯至最大人數，例如: --ramp-max 2000")
    args = parser.parse_args()

    # 決定測試階梯
    if args.ramp_steps:
        stages = sorted(set(args.ramp_steps))
    elif args.ramp_max:
        # 自動產生階梯：500以前加倍，之後每500遞增
        base = [10, 50, 100, 200, 500]
        extra = list(range(1000, args.ramp_max + 1, 500))
        stages = sorted(set(base + extra + [args.ramp_max]))
    elif args.ramp:
        stages = [10, 50, 100, 200, 500, 1000, 2000]
    else:
        stages = [args.users]

    is_multi = len(stages) > 1
    summary  = []

    for n in stages:
        result = await run_stage(n, args.max_tokens, args.timeout)
        if result:
            summary.append(result)
        if is_multi and result and result["fail"] > result["success"]:
            print(f"\n⚠️  失敗率過高（{result['fail']}/{n} 失敗），停止測試。")
            break

    if is_multi and len(summary) > 1:
        print(f"\n{'='*60}")
        print(f"📈 階梯測試總覽")
        print(f"{'─'*60}")
        print(f"{'人數':>6} {'成功':>6} {'失敗':>6} {'吞吐(tok/s)':>12} {'p50(s)':>8} {'p95(s)':>8} {'p99(s)':>8}")
        print(f"{'─'*60}")
        for r in summary:
            print(f"{r['users']:>6} {r['success']:>6} {r['fail']:>6} "
                  f"{r['throughput']:>12,.0f} {r['p50']:>8.2f} {r['p95']:>8.2f} {r['p99']:>8.2f}")
        print(f"{'='*60}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ 測試中斷")
