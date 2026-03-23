import requests

def main():
    r = requests.post('http://127.0.0.1:8000/api/auth/register', json={'email':'probe@example.com','password':'pass123'})
    print('status', r.status_code)
    print(r.text)

if __name__ == '__main__':
    main()
