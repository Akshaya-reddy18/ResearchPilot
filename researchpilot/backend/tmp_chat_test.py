import json
import urllib.request

data = json.dumps({
    'query': 'Summarize agentic AI',
    'top_k': 3,
    'use_context': True
}).encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/v1/chat/chat',
    data=data,
    headers={'Content-Type': 'application/json'},
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode('utf-8')
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2))
except Exception as e:
    print('ERROR', str(e))
    raise
