import os
import sys
from pathlib import Path

print("=== LLM Connectivity Test ===")
print("exe=", sys.executable)

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Robust .env loading (handles UTF-16/BOM)
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    try:
        load_dotenv(dotenv_path=str(env_path), override=True)
    except UnicodeDecodeError:
        try:
            raw = env_path.read_text(encoding="utf-16")
        except Exception:
            raw = env_path.read_text(errors="ignore")
        for line in raw.splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip().lstrip("\ufeff")] = v.strip()

# Show key presence and library versions
from importlib import import_module
try:
    openai_mod = import_module("openai")
    import httpx  # type: ignore
    print("openai=", getattr(openai_mod, "__version__", "?"), "httpx=", getattr(httpx, "__version__", "?"))
except Exception as e:
    print("[WARN] Failed to import libs:", type(e).__name__, e)

has_key = bool(os.getenv("OPENAI_API_KEY"))
print("HAS_KEY=", has_key)
if has_key:
    print("PREFIX=", os.getenv("OPENAI_API_KEY")[:5])

from openai import OpenAI

try:
    # Prefer env var style to avoid proxies kwarg compatibility issues
    client = OpenAI()
    # Test: list models (no cost)
    models = client.models.list()
    print("MODEL_COUNT=", len(models.data))
    # Tiny chat call (few tokens)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=5,
            temperature=0.0,
        )
        print("CHAT_OK=", bool(resp.choices[0].message.content))
    except Exception as e:
        print("CHAT_ERR:", type(e).__name__, e)
    print("RESULT= OK")
except Exception as e:
    print("ERR:", type(e).__name__, e)
    print("HINT: try 'python -m pip install -U openai httpx httpcore' or clear proxies")

