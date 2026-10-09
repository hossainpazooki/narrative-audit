"""Test configuration: put src/ on the import path.

The pipeline modules are scripts, not an installed package, and each one
inserts src/ into sys.path when run directly. Tests import them as modules
instead, so src/ has to be on the path before collection.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
