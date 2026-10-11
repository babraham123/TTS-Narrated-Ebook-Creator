#!/usr/bin/env python3
"""Compatibility entry point; the implementation lives in the calliope package (src/calliope)."""

import sys

from calliope.cli import main

if __name__ == "__main__":
    sys.exit(main())
