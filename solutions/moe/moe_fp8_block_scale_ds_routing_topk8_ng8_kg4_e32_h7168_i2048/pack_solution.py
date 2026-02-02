"""
Pack MoE solution variants into solution JSON files.

Reads config.toml and packs source files for each variant (or a specific one)
into solution JSON files compatible with flashinfer-bench.

Usage:
    python pack_solution.py --all              # Pack all variants
    python pack_solution.py --variant sglang_v1  # Pack a specific variant
"""

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files

SCRIPT_DIR = Path(__file__).parent


def load_config() -> dict:
    config_path = SCRIPT_DIR / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_variant(common: dict, variant_key: str, variant_cfg: dict) -> Path:
    source_dir = SCRIPT_DIR / variant_cfg["source_dir"]
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    spec = BuildSpec(
        language=variant_cfg["language"],
        target_hardware=["NVIDIA B200"],
        entry_point=variant_cfg["entry_point"],
        dependencies=variant_cfg.get("dependencies", []),
        destination_passing_style=variant_cfg.get("destination_passing_style", True),
    )

    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=variant_cfg["name"],
        definition=common["definition"],
        author=common["author"],
        description=variant_cfg.get("description", ""),
    )

    output_path = SCRIPT_DIR / f"{variant_cfg['name']}.json"
    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Packed {variant_key}: {output_path.name}")
    print(f"  Sources: {[s.path for s in solution.sources]}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pack MoE solution variants")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Pack all variants")
    group.add_argument("--variant", type=str, help="Pack a specific variant by key")
    args = parser.parse_args()

    config = load_config()
    common = config["common"]
    variants = config["variants"]

    if args.all:
        keys = list(variants.keys())
    else:
        if args.variant not in variants:
            print(f"Error: Unknown variant '{args.variant}'", file=sys.stderr)
            print(f"Available: {list(variants.keys())}", file=sys.stderr)
            sys.exit(1)
        keys = [args.variant]

    for key in keys:
        try:
            pack_variant(common, key, variants[key])
        except Exception as e:
            print(f"Error packing {key}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
