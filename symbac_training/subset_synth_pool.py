#!/usr/bin/env python3
"""Sample a fixed-size subset from a converted synthetic pool (synth_to_coco.py
output), for synthetic:real ratio ablation (0x/1x/3x etc.) without having to
generate separate pools per ratio. Symlinks rather than copies -- cheap to
build many subsets from one pool.

Usage:
  python subset_synth_pool.py --format mtb_crops --input_dir data/synth_tb_coco \\
      --n 2400 --output_dir data/synth_tb_coco_1x --seed 0
  python subset_synth_pool.py --format pa_records --input_dir data/synth_pa_coco \\
      --n 200 --output_dir data/synth_pa_coco_1x --seed 0
"""

import argparse
import json
import random
from pathlib import Path


def subset_mtb_crops(input_dir, output_dir, n, seed):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(f.stem.replace("_coco", "") for f in input_dir.glob("*_coco.json"))
    if n < len(stems):
        random.Random(seed).shuffle(stems)
        stems = stems[:n]
    elif n > len(stems):
        print(f"WARNING: requested {n} but pool only has {len(stems)} -- using all of them")

    for stem in stems:
        for suffix in (f"{stem}.tiff", f"{stem}_coco.json"):
            src = input_dir / suffix
            dst = output_dir / suffix
            if not dst.exists():
                dst.symlink_to(src.resolve())
    print(f"Subset: {len(stems)} crops -> {output_dir}")


def subset_pa_records(input_dir, output_dir, n, seed):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    with open(input_dir / "records.json") as f:
        records = json.load(f)

    if n < len(records):
        random.Random(seed).shuffle(records)
        records = records[:n]
    elif n > len(records):
        print(f"WARNING: requested {n} but pool only has {len(records)} -- using all of them")

    for rec in records:
        src = input_dir / "images" / rec["file_name"]
        dst = output_dir / "images" / rec["file_name"]
        if not dst.exists():
            dst.symlink_to(src.resolve())
    with open(output_dir / "records.json", "w") as f:
        json.dump(records, f)
    print(f"Subset: {len(records)} records -> {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["mtb_crops", "pa_records"], required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.format == "mtb_crops":
        subset_mtb_crops(args.input_dir, args.output_dir, args.n, args.seed)
    else:
        subset_pa_records(args.input_dir, args.output_dir, args.n, args.seed)


if __name__ == "__main__":
    main()
