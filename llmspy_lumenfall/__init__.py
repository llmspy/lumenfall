"""
Lumenfall extension for llmspy
Lumenfall is an AI media generation gateway with a unified OpenAI-compatible API.

Registers Lumenfall as an image generation and editing provider in llmspy,
giving users access to all top AI image models across all leading providers
with a single API key.

Architecture:
- install() creates a factory that injects ctx into LumenfallImageGenerator
  (defined in generator.py) and stores it as _generator_factory for the
  provider to pick up.
- LumenfallProvider lives in provider.py (importable for tests) and wires
  the generator via _generator_factory in its __init__.
- __load__ auto-registers into g_handlers for zero-config experience.
"""

__version__ = "0.1.0"

# Set by install(), read by provider.py to wire the image modality.
# None when imported standalone (tests), class when running inside llmspy.
_generator_factory = None


def install(ctx):
    import os

    from .models import set_cache_dir

    cache_dir = os.path.join(os.path.expanduser("~"), ".llms", "cache")
    set_cache_dir(cache_dir)

    # -- Wire everything together -----------------------------------------

    from .generator import LumenfallImageGenerator

    # Create a ctx-bound subclass so add_provider can instantiate with **kwargs
    class BoundGenerator(LumenfallImageGenerator):
        def __init__(self, **kwargs):
            super().__init__(ctx=ctx, **kwargs)

    BoundGenerator.sdk = LumenfallImageGenerator.sdk

    global _generator_factory
    _generator_factory = BoundGenerator

    from .provider import LumenfallProvider

    ctx.add_provider(LumenfallProvider)
    ctx.add_provider(BoundGenerator)


async def load(ctx):
    """Auto-register Lumenfall if LUMENFALL_API_KEY is set and no
    explicit config entry exists in llms.json.

    Called after init_llms() populates g_handlers, so we can inject
    directly into the provider registry.
    """
    import os

    providers = ctx.get_providers()
    if "lumenfall" in providers:
        return  # already configured via llms.json

    api_key = os.environ.get("LUMENFALL_API_KEY")
    if not api_key:
        return  # no key, skip

    from .provider import LumenfallProvider

    try:
        provider = LumenfallProvider(api_key=api_key)
        providers["lumenfall"] = provider
        ctx.log("Lumenfall auto-registered (LUMENFALL_API_KEY found)")
    except Exception as e:
        ctx.err("Failed to auto-register Lumenfall provider", e)

    # Async refresh of model catalog
    import aiohttp

    try:
        base_url = os.environ.get(
            "LUMENFALL_BASE_URL", "https://api.lumenfall.ai/openai/v1"
        )
        url = base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    from .models import save_models, get_models

                    save_models(data)
                    # Update the live provider's model list
                    if "lumenfall" in providers:
                        providers["lumenfall"].set_models(models=get_models())
    except Exception:
        pass  # silent fail - static/cached catalog still works


__install__ = install
__load__ = load
