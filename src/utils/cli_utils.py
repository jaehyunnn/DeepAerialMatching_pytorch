"""Command-line interface utilities."""
from __future__ import annotations

import argparse


def str_to_bool(value: str) -> bool:
    """Convert string to boolean for argparse.

    Args:
        value: String representation of boolean.

    Returns:
        Boolean value.

    Raises:
        argparse.ArgumentTypeError: If value cannot be parsed as boolean.
    """
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value expected, got '{value}'"
        )
