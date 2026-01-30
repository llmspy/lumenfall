"""Unit tests for the dynamic model catalog (models.py)."""

from llmspy_lumenfall.models import _parse_models, get_models, save_models, set_cache_dir


class TestParseModels:
    def test_extracts_name(self):
        """name field is used when present."""
        resp = {"data": [{"id": "flux.2-pro", "name": "FLUX 2 Pro", "object": "model"}]}
        models = _parse_models(resp)
        assert models["flux.2-pro"]["name"] == "FLUX 2 Pro"

    def test_falls_back_to_id(self):
        """When name is missing, id is used as fallback."""
        resp = {"data": [{"id": "flux.2-pro", "object": "model"}]}
        models = _parse_models(resp)
        assert models["flux.2-pro"]["name"] == "flux.2-pro"

    def test_empty_data(self):
        """Empty data array returns empty dict."""
        assert _parse_models({"data": []}) == {}

    def test_missing_data_key(self):
        """Missing data key returns empty dict."""
        assert _parse_models({}) == {}


class TestCacheFlow:
    def test_save_and_load(self, tmp_path):
        """save_models() writes JSON that get_models() can read back."""
        set_cache_dir(str(tmp_path))
        api_resp = {"data": [{"id": "mock-image", "name": "Mock Image", "object": "model"}]}
        save_models(api_resp)
        models = get_models()
        assert "mock-image" in models
        assert models["mock-image"]["name"] == "Mock Image"

    def test_static_fallback(self, tmp_path):
        """get_models() falls back to static models.json when no cache exists."""
        set_cache_dir(str(tmp_path / "empty"))  # non-existent cache dir (will be created)
        models = get_models()
        assert len(models) > 0
        assert "mock-image" in models
