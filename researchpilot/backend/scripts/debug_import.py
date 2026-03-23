import os
import sys
import traceback
import importlib

# Ensure backend root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
importlib.invalidate_caches()
try:
    import main
    print('main imported')
except Exception:
    traceback.print_exc()
    sys.exit(1)
