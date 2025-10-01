import sys, os
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from app.config import settings
    print("exe=", sys.executable)
    print("cwd=", os.getcwd())
    print("project_root=", PROJECT_ROOT)
    print("KEY_LOADED=", bool(settings.openai_api_key))
    print("PREFIX=", (settings.openai_api_key or '')[:5])
except Exception as e:
    import traceback
    print("ERROR:", type(e).__name__, e)
    traceback.print_exc()

