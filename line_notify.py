import requests
import argparse
url = 'https://notify-api.line.me/api/notify'
token = 'M1UGrN6zBHLDOuhNoUUkt3ILcSH09o3emCgLr8KU5dy'
headers = {
    'Authorization': 'Bearer '+token  # 設定權杖
}

def send_message(message:str):
    data = {
        "message":message    # 設定要發送的訊息
    }
    data = requests.post(url, headers=headers, data=data)  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message', type=str)
    args = parser.parse_args()
    send_message( args.message)    