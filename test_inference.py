import openai
import sys
import time
import urllib.request
import urllib.error

def wait_for_server(host, port, timeout=600, interval=15):
    """等待 SGLang 伺服器就緒（輪詢 /health 端點）"""
    health_url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    print(f"等待伺服器啟動 ({health_url})，最長等待 {timeout} 秒...")
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[✓] 伺服器就緒！（第 {attempt} 次嘗試）")
                    return True
        except Exception:
            elapsed = int(time.time() - (deadline - timeout))
            print(f"  [{elapsed:4d}s] 伺服器尚未就緒，{interval} 秒後重試...")
            time.sleep(interval)
    print(f"[✗] 逾時：伺服器在 {timeout} 秒內未啟動")
    return False

def get_node_ip():
    """取得本機對外 IP（用於叢集節點確認）"""
    import socket
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "unknown"

def test_sglang(port, model_name, wait=True):
    # 嘗試不同的 host 地址，因為叢集網路綁定方式可能不同
    hosts = ["127.0.0.1", "localhost"]
    success = False
    ready_host = None

    # 先等待伺服器就緒
    if wait:
        for host in hosts:
            if wait_for_server(host, port, timeout=600, interval=15):
                ready_host = host
                break
        if ready_host is None:
            print(f"[失敗] 伺服器未在任何地址就緒，放棄測試。")
            print(f"  節點 IP: {get_node_ip()}")
            return

    test_hosts = [ready_host] if ready_host else hosts
    for host in test_hosts:
        url = f"http://{host}:{port}/v1"
        print(f"\n嘗試連線到 {url} (模型: {model_name})...")
        client = openai.OpenAI(base_url=url, api_key="EMPTY")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "用一句話介紹你自己"}],
                max_tokens=256,
                timeout=120
            )
            print(f"[✓ 測試成功] 透過 {host} 連線！")
            print(f"回應：{response.choices[0].message.content}")
            success = True
            break
        except Exception as e:
            print(f"[✗] 透過 {host} 連線失敗：{e}")

    if not success:
        print(f"\n[所有連線皆失敗]")
        print(f"  節點 IP: {get_node_ip()}")
        print(f"  建議：確認 sglang 是否在 port {port} 上啟動，")
        print(f"        或嘗試直接在計算節點執行 curl http://127.0.0.1:{port}/health")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SGLang MiniMax 推論測試")
    parser.add_argument("--port", type=int, default=8000, help="SGLang 服務 port（預設 8000）")
    parser.add_argument("--model", type=str, default="MiniMaxAI/MiniMax-M2.7", help="模型名稱")
    parser.add_argument("--no-wait", action="store_true", help="不等待伺服器啟動，直接測試")
    args = parser.parse_args()

    test_sglang(args.port, args.model, wait=not args.no_wait)
