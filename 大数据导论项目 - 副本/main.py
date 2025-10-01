from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root BEFORE importing settings
_project_root = Path(__file__).resolve().parent
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
    try:
        load_dotenv(override=True)
    except UnicodeDecodeError:
        pass

from app.config import settings
from app.graph import build_workflow, PipelineState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Used car EDA + Clustering + LLM Report")
    parser.add_argument("--input_csv", type=str, default=settings.input_path, help="Input CSV/XLSX path")
    parser.add_argument("--output_md", type=str, default=settings.output_md, help="Output markdown path")
    parser.add_argument("--img_dir", type=str, default=settings.img_dir, help="Image output directory")
    parser.add_argument("--random_seed", type=int, default=settings.random_seed, help="Random seed for clustering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.img_dir, exist_ok=True)

    graph = build_workflow(
        raw_path=args.input_csv,
        out_md=args.output_md,
        img_dir=args.img_dir,
        random_seed=args.random_seed,
    ).compile()

    state = PipelineState(
        raw_path=args.input_csv,
        output_md=args.output_md,
        img_dir=args.img_dir,
        random_seed=args.random_seed,
    )

    graph.invoke(state)
    print(f"Report generated at: {args.output_md}")


if __name__ == "__main__":
    main()
