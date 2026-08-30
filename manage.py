#!/usr/bin/env python
"""Django management entrypoint for work inside a checkout.

The real entry point is the `openoutfind` console script
(`openoutfind/__main__.py`); this shim exists so `python manage.py <verb>`
keeps working without an install.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openoutfind.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
