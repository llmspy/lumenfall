"""
Lumenfall provider for llmspy.

Extends OpenAiCompatible for proper model resolution, modality dispatch,
header management, and metadata augmentation.

This module is importable standalone (for tests) — the image generator
modality is wired in by __init__.py's install() hook via _generator_factory.
"""

import os

from llms.main import OpenAiCompatible

from .models import get_models


class LumenfallProvider(OpenAiCompatible):
    """llmspy provider for Lumenfall's unified image generation API.

    Uses OpenAiCompatible's provider_model() for model resolution, including
    map_models, prefix stripping, and case-insensitive matching.
    """

    sdk = "llmspy_lumenfall"
    _models_cache = None

    def __init__(self, **kwargs):
        if "api" not in kwargs:
            kwargs["api"] = os.environ.get(
                "LUMENFALL_BASE_URL",
                "https://api.lumenfall.ai/openai/v1",
            )
        if "id" not in kwargs:
            kwargs["id"] = "lumenfall"
        if "env" not in kwargs:
            kwargs["env"] = ["LUMENFALL_API_KEY"]
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.environ.get("LUMENFALL_API_KEY")
        if not kwargs.get("models"):
            kwargs["models"] = get_models()

        super().__init__(**kwargs)

        # Wire up image modality if the generator factory has been set
        # by install(). When imported standalone (e.g. in tests),
        # _generator_factory is None and modalities stay empty — that's
        # fine because tests only need provider_model().
        from . import _generator_factory

        if _generator_factory is not None:
            self.modalities["image"] = _generator_factory(
                id=self.id,
                api=self.api,
                api_key=self.api_key,
            )

    async def chat(self, chat, context=None):
        """Route chat requests, validating image-only models via /v1/models.

        llmspy's --check sends chat completion requests without modalities,
        which would fail for image-only models. Instead, we validate that
        the model exists via the /v1/models endpoint.
        """
        # If modalities are specified, use normal dispatch (image generation)
        modalities = chat.get("modalities") or []
        if len(modalities) > 0:
            return await super().chat(chat, context=context)

        # For non-modality requests (e.g. --check), validate via /v1/models
        model = self.provider_model(chat.get("model", "")) or chat.get("model", "")

        import aiohttp

        # Cache the models list to avoid repeated API calls
        if LumenfallProvider._models_cache is None:
            models_url = self.api.rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    models_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 401:
                        raise PermissionError("Unauthorized: Invalid API key")
                    if response.status >= 400:
                        text = await response.text()
                        raise RuntimeError(
                            f"API error ({response.status}): {text[:200]}"
                        )
                    data = await response.json()
                    LumenfallProvider._models_cache = {
                        m["id"] for m in data.get("data", [])
                    }

        if model not in LumenfallProvider._models_cache:
            raise ValueError(f"Model not found: {model}")

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"{model} (image-only)",
                }
            }]
        }
