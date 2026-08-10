#!/usr/bin/env python3
"""Process AIME datasets to append reasoning instruction to user prompts.

This script reads AIME dataset JSONL files where prompts are in the format:
{"prompt": [{"role": "user", "content": "..."}], "label": "73"}

It appends "\nPlease reason step by step, and put your final answer within \\boxed{}."
to the user's content to encourage step-by-step reasoning and formatted answers.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def process_prompt(prompt_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append instruction to user role in prompt.

    Args:
        prompt_data: List of message dicts with 'role' and 'content' keys

    Returns:
        Modified prompt data with instruction appended to user content
    """
    for message in prompt_data:
        if message.get("role") == "user":
            original_content = message.get("content", "")
            message["content"] = original_content + INSTRUCTION
    return prompt_data


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    backup: bool = True,
) -> None:
    """Process a single JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        backup: If True and input==output, create backup first
    """
    if input_path == output_path and backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".backup")
        print(f"Creating backup: {backup_path}")
        backup_path.write_text(input_path.read_text())

    processed_count = 0
    error_count = 0

    new_data = []
    with open(input_path, "r") as f_in:
        for line in f_in.readlines():
            data = json.loads(line)
            data["prompt"] = process_prompt(data["prompt"])
            processed_count += 1
            new_data.append(data)
    with open(output_path, "w") as f_out:
        for data in new_data:
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\nProcessed {input_path}:")
    print(f"  - Lines processed: {processed_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process AIME datasets to append reasoning instruction"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="Input JSONL file(s) to process",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input, creates backup)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup when modifying in place",
    )

    args = parser.parse_args()

    for input_file in args.input_files:
        if not input_file.exists():
            print(f"Warning: File not found: {input_file}")
            continue

        if args.output_dir:
            output_dir = args.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / input_file.name.replace(".jsonl", "_eval.jsonl")
        else:
            output_file = input_file

        process_jsonl_file(
            input_file,
            output_file,
            backup=not args.no_backup,
        )


if __name__ == "__main__":
    main()
