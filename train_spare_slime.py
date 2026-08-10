#!/usr/bin/env python3
"""SPARE training wrapper for Slime.

This script wraps Slime's training loop with SPARE-specific argument handling.
It uses Slime's extension points to add SPARE arguments without modifying Slime's core.
"""

from slime.train import train


def main():
    """Parse arguments with SPARE extensions and run training."""
    from slime.utils.arguments import parse_args

    try:
        from spare.slime.arguments import add_spare_arguments

        args = parse_args(add_custom_arguments=add_spare_arguments)
    except ImportError as e:
        print(f"Warning: SPARE arguments not available ({e}), using default Slime args")
        args = parse_args()

    train(args)


if __name__ == "__main__":
    main()
