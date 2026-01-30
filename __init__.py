"""
Bridge for llms --add installation.

When installed via ``llms --add lumenfall-ai/llmspy-lumenfall``, the repo
root becomes ``~/.llms/extensions/llmspy-lumenfall/``.  llmspy's extension
loader looks for ``__init__.py`` at that level, so this file re-exports
the hooks from the actual package subdirectory.
"""

import sys
import os

# Add the repo root to sys.path so the llmspy_lumenfall package is importable.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from llmspy_lumenfall import __install__, __load__
