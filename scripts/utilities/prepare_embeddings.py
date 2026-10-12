#!/usr/bin/env python3
"""Precompute BF16 text embeddings in the format consumed by training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from utils.wan_wrapper import WanTextEncoder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output: {args.output_dir}")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")

    prompts = args.prompts.read_text(encoding="utf-8").splitlines()
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompt file must contain non-empty, one-line prompts")

    args.output_dir.mkdir(parents=True)
    selected_prompts = args.output_dir / "selected_prompts.txt"
    shutil.copyfile(args.prompts, selected_prompts)

    encoder = WanTextEncoder().to(device=args.device, dtype=torch.bfloat16)
    encoder.eval()
    records: list[dict[str, object]] = []
    shard: dict[str, torch.Tensor] = {}
    shard_id = 0

    def flush() -> None:
        nonlocal shard, shard_id
        if not shard:
            return
        name = f"embeddings-{shard_id:05d}.safetensors"
        save_file(shard, str(args.output_dir / name))
        shard = {}
        shard_id += 1

    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            encoded = encoder(text_prompts=[prompt])["prompt_embeds"]
            tensor = encoded[0].detach().to(device="cpu", dtype=torch.bfloat16)
            key = f"prompt_{index:08d}"
            shard[key] = tensor.contiguous()
            records.append(
                {
                    "index": index,
                    "shard": f"embeddings-{shard_id:05d}.safetensors",
                    "key": key,
                    "length": tensor.shape[0],
                }
            )
            if len(shard) == args.shard_size:
                flush()
        flush()

        negative = encoder(text_prompts=[args.negative_prompt])["prompt_embeds"]
        negative = negative[0].detach().to(device="cpu", dtype=torch.bfloat16)
        save_file(
            {"prompt_embeds": negative.contiguous()},
            str(args.output_dir / "negative_prompt.safetensors"),
        )

    with (args.output_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "prospective_forcing_embeddings_v1",
        "dtype": "bfloat16",
        "selected_prompt_count": len(prompts),
        "prompt_sha256": sha256(selected_prompts),
        "shard_size": args.shard_size,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
