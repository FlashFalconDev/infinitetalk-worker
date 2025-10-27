"""
測試：找出為什麼沒有響應
"""
import requests
import traceback

url = "https://www.flashfalcon.info/ai/api/pending_task/"
token = "hwDblsQ707fWmkue2amxIRShIXplI58Bl0nJXxVLKmo"

headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

params = {'model_code': 'InfiniteTalk_S2V'}

print("=" * 70)
print(f"測試請求: GET {url}")
print(f"Token: {token[:10]}...{token[-10:]}")
print(f"Headers: {headers}")
print(f"Params: {params}")
print("=" * 70)

try:
    print("\n開始請求...")
    response = requests.get(url, params=params, headers=headers, timeout=30)
    
    print(f"\n✅ 有響應!")
    print(f"狀態碼: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"\n響應內容:")
    print(response.text)
    
except requests.exceptions.Timeout as e:
    print(f"\n❌ Timeout: {e}")
    traceback.print_exc()
    
except requests.exceptions.ConnectionError as e:
    print(f"\n❌ ConnectionError: {e}")
    traceback.print_exc()
    
except requests.exceptions.RequestException as e:
    print(f"\n❌ RequestException: {e}")
    traceback.print_exc()
    
except Exception as e:
    print(f"\n❌ 未知錯誤: {type(e).__name__} - {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
