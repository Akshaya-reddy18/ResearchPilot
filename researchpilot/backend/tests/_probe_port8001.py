import requests

def main():
    url = 'http://127.0.0.1:8001/api/auth/register'
    payload = {'email': 'probe_port8001@example.com', 'password': 'pass123'}
    r = requests.post(url, json=payload)
    print('STATUS', r.status_code)
    print(r.text)

if __name__ == '__main__':
    main()
