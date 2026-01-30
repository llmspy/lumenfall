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
