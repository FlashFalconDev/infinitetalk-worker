import requests
import os
import json

def upload_file(file_path, upload_url="https://host.flashfalcon.info/api/save_file/"):
    """測試上傳文件"""
    if not os.path.exists(file_path):
        print(f"錯誤: 文件不存在 {file_path}")
        return None
    
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
                    return None
            else:
                print(f"❌ HTTP 錯誤: {response.status_code}")
                return None
                
    except requests.exceptions.RequestException as e:
        print(f"❌ 網路錯誤: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析錯誤: {e}")
        return None
    except Exception as e:
        print(f"❌ 未知錯誤: {e}")
        return None

if __name__ == "__main__":
    # 上傳剛剛生成的測試影片
    video_url = upload_file("test_service_output.mp4")
    
    if video_url:
        print(f"\n✅ 可以使用的影片網址:")
        print(video_url)
    else:
        print("\n❌ 上傳失敗")
