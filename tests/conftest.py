"""
Pytest configuration and fixtures for Lumenfall llmspy extension tests.

Manages extension installation into llmspy's runtime and provides helpers
for running CLI commands and locating generated images.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTENSION_NAME = "llmspy_lumenfall"

# The extension source package (one level up from tests/)
EXTENSION_SRC = Path(__file__).resolve().parent.parent / EXTENSION_NAME

# Where llmspy looks for user-installed extensions
LLMS_EXTENSIONS_DIR = Path.home() / ".llms" / "extensions"

# llmspy caches images here by default (overridable via LLMS_HOME)
LLMS_CACHE_DIR = Path.home() / ".llms" / "cache"

API_KEY = os.environ.get("LUMENFALL_API_KEY")

# Model used for e2e tests.  Default is mock-image (free, server-side test model).
# Override with LUMENFALL_TEST_MODEL=flux.2-pro for real-world validation.
TEST_MODEL = os.environ.get("LUMENFALL_TEST_MODEL", "mock-image")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_llms(*args, env_override=None, timeout=60):
    """
    Run the ``llms`` CLI as a subprocess and return CompletedProcess.

    All output is captured as text.  *env_override* merges into the current
    environment so the caller can set things like LUMENFALL_API_KEY or
    LLMS_CONFIG_PATH without clobbering the rest.
    """
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    return subprocess.run(
        ["llms", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def run_extension_script(script_body, timeout=15):
    """
    Execute a short Python snippet with the extension package on sys.path.

    Useful for testing provider/generator internals without going through the
    full llmspy runtime.  Automatically sets LUMENFALL_API_KEY if available.
    """
    preamble = (
        "import sys, os; "
        f"sys.path.insert(0, {str(EXTENSION_SRC.parent)!r}); "
        f"os.environ.setdefault('LUMENFALL_API_KEY', {(API_KEY or 'test')!r}); "
    )
    return subprocess.run(
        ["python", "-c", preamble + script_body],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def find_saved_files(stdout):
    """
    Parse llmspy's CLI output to extract saved file paths.

    llmspy prints a "Saved files:" header followed by alternating lines of
    absolute local paths and localhost HTTP URLs.  This function returns the
    local paths only.
    """
    paths = []
    in_saved_section = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("saved files"):
            in_saved_section = True
            continue
        if in_saved_section and stripped:
            # Skip the localhost HTTP URL lines
            if stripped.startswith("http://") or stripped.startswith("https://"):
                continue
            # Anything else in this section should be an absolute file path
            if stripped.startswith("/"):
                paths.append(Path(stripped))
    return paths


def find_cached_images(after=0):
    """
    Return image files from llmspy's cache dir modified at or after *after*.

    Sorted newest-first.
    """
    if not LLMS_CACHE_DIR.is_dir():
        return []
    return sorted(
        (
            f for f in LLMS_CACHE_DIR.rglob("*")
            if f.is_file()
            and f.suffix in (".png", ".jpg", ".jpeg", ".webp")
            and f.stat().st_mtime >= after
        ),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )


def is_valid_image(path):
    """Return True if *path* begins with PNG or JPEG magic bytes."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
        return header[:4] == b"\x89PNG" or header[:2] == b"\xff\xd8"
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def extension_dir():
    """
    Symlink the extension source into ~/.llms/extensions/ for the test session.

    Tears down the symlink afterwards unless it existed before we started.
    """
    target = LLMS_EXTENSIONS_DIR / EXTENSION_NAME
    LLMS_EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)

    already_existed = target.exists() or target.is_symlink()

    if not already_existed:
        target.symlink_to(EXTENSION_SRC)

    yield target

    if not already_existed and target.is_symlink():
        target.unlink()


@pytest.fixture(scope="session")
def api_key():
    """Return the Lumenfall API key from the environment."""
    return API_KEY
