"""
End-to-end tests for the Lumenfall llmspy extension.

All tests default to the ``mock-image`` model (a free, server-side test model)
so they can run without spending API credits.

Requirements:
    - ``llms-py`` installed (``pip install llms-py``)
    - ``LUMENFALL_API_KEY`` environment variable set
    - Extension source at ``../llmspy_lumenfall/``

To run against a real model instead of the mock:

    LUMENFALL_TEST_MODEL=flux.2-pro pytest tests/test_e2e.py
"""

import json
import shutil
import subprocess
import time

import pytest

from conftest import (
    API_KEY,
    EXTENSION_NAME,
    EXTENSION_SRC,
    LLMS_EXTENSIONS_DIR,
    TEST_MODEL,
    find_cached_images,
    find_saved_files,
    is_valid_image,
    run_extension_script,
    run_llms,
)

# ---------------------------------------------------------------------------
# Module-level skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.skipif(
        not shutil.which("llms"),
        reason="llms-py not installed",
    ),
    pytest.mark.skipif(
        not API_KEY,
        reason="LUMENFALL_API_KEY not set",
    ),
]


# ============================================================================
# 1. Extension Installation  (1 test)
# ============================================================================


class TestExtensionInstallation:

    def test_extension_installs(self):
        """Symlink the extension and verify llmspy boots without crashing.

        Validates that __init__.py exists and the __install__ hook doesn't
        raise.  Note: ``llms --version`` may exit non-zero due to unrelated
        extensions (e.g. 'computer' needing a display server), so we only
        check that our extension isn't mentioned in the error output.
        """
        target = LLMS_EXTENSIONS_DIR / EXTENSION_NAME
        LLMS_EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Start from a clean slate
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)

        target.symlink_to(EXTENSION_SRC)

        assert target.exists(), f"Extension not found at {target}"
        assert (target / "__init__.py").exists(), "Extension missing __init__.py"

        result = run_llms("--version")
        # Other extensions may fail (e.g. 'computer' without a display),
        # so we only verify our extension didn't cause an error.
        combined_err = (result.stdout + result.stderr).lower()
        assert "lumenfall" not in combined_err or result.returncode == 0, (
            f"llms crashed due to lumenfall extension.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ============================================================================
# 2. Provider Registration  (2 tests)
# ============================================================================


class TestProviderRegistration:
    """After installation, does llmspy see Lumenfall as a provider with
    the full model catalog?"""

    def test_provider_listed(self, extension_dir):
        """'lumenfall' should appear as a provider heading in ``llms --list``.

        llmspy prints each provider as ``provider_name:`` followed by
        indented model lines.  Finding 'lumenfall' proves add_provider()
        succeeded at registration time.
        """
        result = run_llms("--list")
        assert result.returncode == 0, f"llms --list failed:\n{result.stdout}"

        assert "lumenfall" in result.stdout.lower(), (
            f"'lumenfall' not found in provider listing.\n"
            f"Output (first 1500 chars):\n{result.stdout[:1500]}"
        )

    def test_catalog_includes_real_models(self, extension_dir):
        """At least one flagship model should appear, proving the static
        catalog loaded -- not just the mock model."""
        result = run_llms("--list")
        assert result.returncode == 0

        flagship_models = ["flux.2-pro", "dall-e-3", "gpt-image-1"]
        found = [m for m in flagship_models if m in result.stdout]
        assert found, (
            f"None of {flagship_models} found in catalog -- static model "
            f"list may be broken.\nOutput (first 1500 chars):\n"
            f"{result.stdout[:1500]}"
        )


# ============================================================================
# 3. Image Generation via CLI  (2 tests)
# ============================================================================


class TestImageGeneration:
    """Does the full CLI pipeline produce valid image files on disk?"""

    def test_generate_produces_valid_image_file(self, extension_dir):
        """End-to-end: prompt in, valid image file on disk.

        This is THE core integration test.  It exercises the full pipeline:
        prompt extraction -> chat() -> API call (or mock) -> response
        parsing -> to_response() -> image decoded & cached to disk ->
        llmspy prints the path to stdout.
        """
        before = time.time()
        result = run_llms(
            "--out", "image",
            "A blue square on white background",
            "-m", TEST_MODEL,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Generation failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Primary: parse the "Saved files:" section from stdout
        saved = find_saved_files(result.stdout)

        # Fallback: scan the cache directory for recent image files
        if not saved:
            saved = list(find_cached_images(after=before))

        assert saved, (
            f"Could not locate any generated image file.\n"
            f"stdout:\n{result.stdout}"
        )
        assert is_valid_image(saved[0]), (
            f"File at {saved[0]} does not start with PNG or JPEG magic bytes"
        )

    def test_generate_multiple_images(self, extension_dir):
        """Requesting n=2 should produce at least two distinct image files.

        This exercises the n>1 code path: the generator must return multiple
        entries in message.images, and llmspy must cache each one as a
        separate file.
        """
        before = time.time()
        result = run_llms(
            "--out", "image",
            "Two different patterns",
            "-m", TEST_MODEL,
            "--args", "n=2",
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Generation failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        saved = find_saved_files(result.stdout)
        if not saved:
            saved = list(find_cached_images(after=before))

        assert len(saved) >= 2, (
            f"Expected at least 2 image files, got {len(saved)}.\n"
            f"stdout:\n{result.stdout}"
        )
        assert len(set(saved)) >= 2, (
            f"Expected 2 distinct files, but got duplicates: {saved}"
        )


# ============================================================================
# 4. Response Structure  (1 test)
# ============================================================================


class TestResponseStructure:
    """Does the generator return the response format llmspy expects?"""

    def test_raw_response_has_message_images(self, extension_dir):
        """The --raw JSON response must contain message.images, not image
        data stuffed into message.content.

        This is the critical contract between our generator's to_response()
        and llmspy's image caching layer.  If message.images is missing,
        llmspy won't cache the images and the CLI will print raw base64
        or data URIs instead of file paths.

        Note: ``llms --raw`` may not exit cleanly (hangs after output).
        We handle TimeoutExpired and parse whatever output was produced.
        """
        try:
            result = run_llms(
                "--out", "image",
                "Response structure test",
                "-m", TEST_MODEL,
                "--raw",
                timeout=30,
            )
            stdout = result.stdout
            assert result.returncode == 0, (
                f"Generation failed.\nstdout: {stdout}\nstderr: {result.stderr}"
            )
        except subprocess.TimeoutExpired as exc:
            # llms --raw may hang after producing output; that's OK
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            assert stdout.strip(), (
                "llms --raw timed out without producing any output"
            )

        # Strip llmspy ERROR lines (from other extensions) before parsing JSON
        json_lines = [
            line for line in stdout.splitlines()
            if not line.startswith("ERROR:")
        ]
        data = json.loads("\n".join(json_lines))

        assert "choices" in data, f"No 'choices' in response: {list(data.keys())}"
        assert len(data["choices"]) >= 1, "Empty choices array"

        msg = data["choices"][0]["message"]
        assert msg.get("role") == "assistant", (
            f"message.role is {msg.get('role')!r}, expected 'assistant'"
        )

        # THE key assertion: images must be in message.images
        assert "images" in msg, (
            f"message.images is missing -- the generator likely returns the "
            f"wrong response format.  Got message keys: {list(msg.keys())}"
        )
        assert len(msg["images"]) >= 1, "message.images is empty"

        # Each image entry should have the expected shape
        for img in msg["images"]:
            assert "image_url" in img, f"Image entry missing 'image_url': {img}"
            assert "url" in img["image_url"], (
                f"Image entry missing 'image_url.url': {img}"
            )


# ============================================================================
# 5. Error Handling  (1 test)
# ============================================================================


class TestErrorHandling:
    """Are errors surfaced as readable messages, not raw stack traces?

    Note: llmspy prints error messages to stdout, not stderr.
    """

    def test_invalid_api_key(self, extension_dir):
        """An invalid API key should produce a non-zero exit code and an
        error message mentioning authentication."""
        result = run_llms(
            "--out", "image",
            "This should fail",
            "-m", TEST_MODEL,
            env_override={"LUMENFALL_API_KEY": "invalid_key_12345"},
            timeout=30,
        )
        assert result.returncode != 0, "Should fail with invalid API key"

        # llmspy prints errors to stdout
        combined = (result.stdout + result.stderr).lower()
        auth_indicators = [
            "auth", "unauthorized", "401", "invalid", "api key",
            "forbidden", "permission",
        ]
        assert any(w in combined for w in auth_indicators), (
            f"Error output doesn't mention authentication.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ============================================================================
# 6. Configuration  (1 test)
# ============================================================================


class TestConfiguration:
    """Do configuration overrides take effect?"""

    def test_custom_base_url_is_used(self, extension_dir):
        """LUMENFALL_BASE_URL should override the default API endpoint.

        We point it at an unreachable address so the connection-refused
        error proves the override took effect.
        """
        result = run_llms(
            "--out", "image",
            "This should fail",
            "-m", TEST_MODEL,
            env_override={"LUMENFALL_BASE_URL": "http://localhost:19999"},
            timeout=30,
        )
        assert result.returncode != 0, "Should fail with unreachable URL"

        combined = (result.stdout + result.stderr).lower()
        connection_indicators = [
            "connect", "refused", "timeout", "unreachable",
            "error", "failed",
        ]
        assert any(w in combined for w in connection_indicators), (
            f"Error doesn't reference connection failure.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ============================================================================
# 7. Model Resolution  (2 tests)
# ============================================================================


class TestModelResolution:
    """Does provider_model() correctly claim image models and reject
    everything else?"""

    def test_provider_model_resolution(self, extension_dir):
        """provider_model() should return an ID for known models, strip
        the 'lumenfall/' prefix, and return None for unknowns."""
        result = run_extension_script(
            "from llmspy_lumenfall.provider import LumenfallProvider\n"
            "p = LumenfallProvider()\n"
            # Known model -> returns the model id
            "claimed = p.provider_model('mock-image')\n"
            "assert claimed is not None, "
            "f'provider_model(mock-image) returned {claimed!r}'\n"
            # Unknown model -> returns None
            "rejected = p.provider_model('nonexistent-xyz-99')\n"
            "assert rejected is None, "
            "f'provider_model(nonexistent-xyz-99) returned {rejected!r}'\n"
            # Prefixed model -> strips prefix and resolves
            "prefixed = p.provider_model('lumenfall/flux.2-pro')\n"
            "assert prefixed is not None, "
            "f'provider_model(lumenfall/flux.2-pro) returned {prefixed!r}'\n"
            "print('OK')"
        )
        assert result.returncode == 0, (
            f"provider_model() resolution failed.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_provider_rejects_text_models(self, extension_dir):
        """Lumenfall is an image-only provider and must not claim text
        models.  If provider_model() matches everything, it would
        intercept text chat requests meant for other providers."""
        result = run_extension_script(
            "from llmspy_lumenfall.provider import LumenfallProvider\n"
            "p = LumenfallProvider()\n"
            "text_models = ['gpt-4o', 'claude-3-opus', 'llama-3', "
            "'mistral-large', 'gemini-2.5-flash']\n"
            "for m in text_models:\n"
            "    r = p.provider_model(m)\n"
            "    assert r is None, "
            "f'provider_model({m}) returned {r!r}, should be None'\n"
            "print('OK')"
        )
        assert result.returncode == 0, (
            f"Provider incorrectly claims text models.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
