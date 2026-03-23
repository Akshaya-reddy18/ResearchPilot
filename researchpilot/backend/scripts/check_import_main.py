import traceback
import importlib
import sys
# Ensure backend folder is on sys.path for imports
sys.path.insert(0, r"D:\Research AI\ResearchHub-AI\researchpilot\backend")

try:
    importlib.import_module('main')
    print('IMPORT_OK')
except Exception:
    traceback.print_exc()
    sys.exit(1)
