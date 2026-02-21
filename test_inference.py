import openai
import sys

def test_sglang(port, model_name):
    # 嘗試不同的 host 地址，因為叢集網路綁定方式可能不同
    hosts = ["127.0.0.1", "0.0.0.0", "localhost"]
    success = False
    
    for host in hosts:
        url = f"http://{host}:{port}/v1"
        print(f"嘗試連線到 {url} (模型: {model_name})...")
        client = openai.OpenAI(base_url=url, api_key="EMPTY")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "編寫貪食蛇前端程式碼"}],
                max_tokens=30000,
                timeout=500
            )
            print(f"[測試成功] 透過 {host} 連線成功！回應：{response.choices[0].message.content}")
            success = True
            break
        except Exception as e:
            print(f"透過 {host} 連線失敗")
            
    if not success:
        print(f"\n[所有連線皆失敗] 回報: 伺服器雖然顯示啟動成功，但 API 端點無法存取。")
        print(f"請檢查是否為 140.110.148.3 這種特定 IP。")

if __name__ == "__main__":
    # 預設測試 8000 (MiniMax) 和 30000 (Qwen)
    test_sglang(8000, "MiniMaxAI/MiniMax-M2.5")
    print("-" * 40)
    test_sglang(30000, "Qwen/Qwen3-Coder-Next")
