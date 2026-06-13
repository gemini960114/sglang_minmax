#!/usr/bin/env python3
import os
import sys

def patch_sglang():
    try:
        import sglang
    except ImportError:
        print("[Error] 無法匯入 sglang，請確認已啟用虛擬環境（例如執行 source .venv/bin/activate）")
        sys.exit(1)

    sglang_dir = os.path.dirname(sglang.__file__)
    protocol_path = os.path.join(sglang_dir, "srt", "entrypoints", "anthropic", "protocol.py")
    serving_path = os.path.join(sglang_dir, "srt", "entrypoints", "anthropic", "serving.py")

    # 1. 修改 protocol.py
    if not os.path.exists(protocol_path):
        print(f"[Error] 找不到檔案: {protocol_path}")
        sys.exit(1)

    print(f"正在修補 {protocol_path} ...")
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol_content = f.read()

    target_literal = 'role: Literal["user", "assistant"]'
    replacement_literal = 'role: Literal["user", "assistant", "system", "ctx", "msg"]'

    if target_literal in protocol_content:
        protocol_content = protocol_content.replace(target_literal, replacement_literal)
        with open(protocol_path, "w", encoding="utf-8") as f:
            f.write(protocol_content)
        print("[✓] protocol.py 修補成功！")
    elif replacement_literal in protocol_content:
        print("[o] protocol.py 已經修補過，跳過。")
    else:
        print("[!] 找不到目標特徵，可能 sglang 版本不同。請手動確認。")

    # 2. 修改 serving.py
    if not os.path.exists(serving_path):
        print(f"[Error] 找不到檔案: {serving_path}")
        sys.exit(1)

    print(f"正在修補 {serving_path} ...")
    with open(serving_path, "r", encoding="utf-8") as f:
        serving_content = f.read()

    target_loop = """        # Convert messages
        for msg in anthropic_request.messages:
            if isinstance(msg.content, str):
                openai_messages.append({"role": msg.role, "content": msg.content})
                continue

            # Complex content with blocks
            openai_msg = {"role": msg.role}"""

    replacement_loop = """        # Convert messages
        for msg in anthropic_request.messages:
            role = msg.role
            if role == "ctx":
                role = "system"
            elif role not in ["user", "assistant", "system"]:
                role = "user"

            if isinstance(msg.content, str):
                openai_messages.append({"role": role, "content": msg.content})
                continue

            # Complex content with blocks
            openai_msg = {"role": role}"""

    # 同時要替換後續判斷工具的 role 部分
    target_tool_role = 'if msg.role == "user":'
    replacement_tool_role = 'if role == "user":'

    modified = False
    if target_loop in serving_content:
        serving_content = serving_content.replace(target_loop, replacement_loop)
        modified = True
    
    if target_tool_role in serving_content:
        serving_content = serving_content.replace(target_tool_role, replacement_tool_role)
        modified = True

    if modified:
        with open(serving_path, "w", encoding="utf-8") as f:
            f.write(serving_content)
        print("[✓] serving.py 修補成功！")
    elif "role = msg.role" in serving_content:
        print("[o] serving.py 已經修補過，跳過。")
    else:
        print("[!] 找不到目標特徵，可能 sglang 版本不同。請手動確認。")

    # 同步修補檔至 Singularity 掛載目錄
    import shutil
    project_dir = os.path.dirname(os.path.abspath(__file__))
    patched_dir = os.path.join(project_dir, "patched_anthropic")
    try:
        os.makedirs(patched_dir, exist_ok=True)
        shutil.copy2(protocol_path, os.path.join(patched_dir, "protocol.py"))
        shutil.copy2(serving_path, os.path.join(patched_dir, "serving.py"))
        print(f"[✓] 已同步修補檔至 Singularity 掛載目錄: {patched_dir}")
    except Exception as e:
        print(f"[!] 無法同步修補檔至 Singularity 目錄: {e}")

if __name__ == "__main__":
    patch_sglang()
