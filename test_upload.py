import requests
import os
import json

def upload_file(file_path, upload_url="https://host.flashfalcon.info/api/save_file/"):
    """測試上傳文件"""
    
    if not os.path.exists(file_path):
        print(f"錯誤: 文件不存在 {file_path}")
        return
    
    file_size = os.path.getsize(file_path) / 1024 / 1024
    print(f"文件: {file_path}")
    print(f"大小: {file_size:.2f} MB")
    print(f"上傳至: {upload_url}")
    print("-" * 50)
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "video/mp4")}
            
            print("上傳中...")
            response = requests.post(upload_url, files=files, timeout=600)
            
            print(f"\n狀態碼: {response.status_code}")
            print(f"原始回應:\n{response.text}\n")
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("ok"):
                    print("=" * 50)
                    print("✅ 上傳成功!")
                    print("=" * 50)
                    print(f"URL: {result['data']['url']}")
                    print("=" * 50)
                    return result['data']['url']
                else:
                    print(f"❌ API 返回失敗: {result}")
            else:
                print(f"❌ HTTP 錯誤: {response.status_code}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ 網路錯誤: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析錯誤: {e}")
    except Exception as e:
        print(f"❌ 未知錯誤: {e}")

if __name__ == "__main__":
    # 上傳最近生成的影片
    upload_file("outputs/5dc2d8ef-ac43-4075-8a97-6a327528a7a6_output.mp4")
