import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path


# 1) 先尝试从项目根目录加载 .env（相对于本文件的上一级目录）
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    try:
        load_dotenv(dotenv_path=str(_env_path), override=True)
    except UnicodeDecodeError:
        # Fallback: handle UTF-16/BOM encoded .env manually
        try:
            raw = _env_path.read_text(encoding="utf-16")
        except Exception:
            raw = _env_path.read_text(errors="ignore")
        for line in raw.splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip().lstrip("\ufeff")
                v = v.strip()
                if k:
                    os.environ[k] = v
else:
    # 2) 兜底：从当前工作目录加载
    try:
        load_dotenv(override=True)
    except UnicodeDecodeError:
        # Ignore; no specific path given
        pass

# 额外健壮化：处理 UTF-8 BOM/空格/大小写导致的键名异常
def _find_env_key(target: str) -> Optional[str]:
    target_norm = target.strip().lower()
    for k in os.environ.keys():
        k_norm = k.strip().lower().lstrip("\ufeff")
        if k_norm == target_norm:
            return k
    return None


@dataclass
class Settings:
    # 通过规范化查找键名，兼容可能的 BOM/大小写/空格问题
    _key_name = _find_env_key("OPENAI_API_KEY")
    openai_api_key: str = os.getenv(_key_name, "") if _key_name else os.getenv("OPENAI_API_KEY", "")
    input_path: str = os.getenv("INPUT_PATH", "dataset.xlsx")
    output_md: str = os.getenv("OUTPUT_MD", "report.md")
    img_dir: str = os.getenv("IMG_DIR", "images")
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    input_sheet: Optional[str] = os.getenv("INPUT_SHEET", None)
    # HEADER_ROW: 1-based row number specifying where column names are; empty means auto-detect
    header_row_1based_env: str = os.getenv("HEADER_ROW", "")

    @property
    def header_row_index(self) -> Optional[int]:
        if not self.header_row_1based_env:
            return None
        try:
            val = int(self.header_row_1based_env)
            return max(val - 1, 0)
        except Exception:
            return None


settings = Settings()


