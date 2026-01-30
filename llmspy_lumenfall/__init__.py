"""
Lumenfall extension for llmspy
Lumenfall is an AI media generation gateway with a unified OpenAI-compatible API.

Registers Lumenfall as an image generation provider in llmspy, giving users
access to all top AI image models across all leading providers with a single API key.

Architecture:
- install() defines LumenfallImageGenerator inside the closure (needs ctx
  for save_image_to_cache, last_user_prompt, etc.) and stores it as
  _generator_factory for the provider to pick up.
- LumenfallProvider lives in provider.py (importable for tests) and wires
  the generator via _generator_factory in its __init__.
- __load__ auto-registers into g_handlers for zero-config experience.
"""

__version__ = "0.1.0"

# Set by install(), read by provider.py to wire the image modality.
# None when imported standalone (tests), class when running inside llmspy.
_generator_factory = None


def install(ctx):
    import base64
    import json
    import os
    import time

    import aiohttp

    from llms.main import GeneratorBase

    from .models import set_cache_dir

    cache_dir = os.path.join(os.path.expanduser("~"), ".llms", "cache")
    set_cache_dir(cache_dir)

    # ------------------------------------------------------------------
    # Image Generator (needs ctx closure)
    # ------------------------------------------------------------------

    class LumenfallImageGenerator(GeneratorBase):
        """Image generator backed by Lumenfall's OpenAI-compatible API.

        Extends GeneratorBase to implement:
        - chat() — calls Lumenfall /images/generations
        - to_response() — decodes images, caches to disk via
          ctx.save_image_to_cache, returns message.images with /~cache/ paths
        """

        sdk = "llmspy_lumenfall/image"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.api = kwargs.get("api", "")
            if not self.api.endswith("/images/generations"):
                self.api = self.api.rstrip("/") + "/images/generations"

        # -- API call -----------------------------------------------------

        async def chat(self, chat, provider=None, context=None):
            model = chat.get("model", "")
            started_at = time.time()

            prompt = ctx.last_user_prompt(chat)
            if not prompt:
                raise ValueError("No prompt found in chat messages")

            aspect_ratio = ctx.chat_to_aspect_ratio(chat) or "1:1"

            payload = {
                "model": model,
                "prompt": prompt,
                "n": chat.get("n", 1),
                "response_format": "b64_json",
                "aspect_ratio": aspect_ratio,
            }

            headers = self.get_headers(provider, chat)

            ctx.log(f"POST {self.api}")
            ctx.log(json.dumps(payload, indent=2))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    text = await response.text()
                    ctx.log(
                        text[:1024] + ("..." if len(text) > 1024 else "")
                    )

                    if response.status == 401:
                        raise PermissionError(
                            "Unauthorized: Invalid API key. "
                            "Check LUMENFALL_API_KEY environment variable."
                        )
                    if response.status == 404:
                        raise ValueError(f"Model not found: {model}")
                    if response.status >= 400:
                        try:
                            err = json.loads(text)
                            msg = err.get("error", {}).get("message", text)
                        except (ValueError, KeyError):
                            msg = text
                        raise RuntimeError(
                            f"API error ({response.status}): {msg}"
                        )

                    return ctx.log_json(
                        await self.to_response(
                            json.loads(text), chat, started_at
                        )
                    )

        # -- Response processing ------------------------------------------

        async def to_response(self, response, chat, started_at, context=None):
            """Decode images, cache to disk, return llmspy-format response.

            The response must have message.images with /~cache/ URLs so
            llmspy's CLI prints "Saved files:" with local paths.
            """
            if "error" in response:
                raise RuntimeError(
                    response["error"].get("message", str(response["error"]))
                )

            data = response.get("data")
            if not data:
                ctx.log(json.dumps(response, indent=2))
                raise RuntimeError("No 'data' field in API response")

            images = []
            for i, item in enumerate(data):
                b64_data = item.get("b64_json")
                image_url = item.get("url")

                ext = "png"
                image_bytes = None

                if b64_data:
                    image_bytes = base64.b64decode(b64_data)
                elif image_url:
                    ctx.log(f"GET {image_url}")
                    async with aiohttp.ClientSession() as dl_session:
                        async with dl_session.get(image_url) as res:
                            if res.status == 200:
                                image_bytes = await res.read()
                                ct = res.headers.get("Content-Type", "")
                                if "jpeg" in ct or "jpg" in ct:
                                    ext = "jpg"
                                elif "webp" in ct:
                                    ext = "webp"
                            else:
                                raise RuntimeError(
                                    f"Failed to download image: "
                                    f"HTTP {res.status}"
                                )

                if image_bytes:
                    relative_url, _info = ctx.save_image_to_cache(
                        image_bytes,
                        f"{chat.get('model', 'image')}-{i}.{ext}",
                        ctx.to_file_info(chat),
                    )
                    images.append({
                        "type": "image_url",
                        "image_url": {"url": relative_url},
                    })
                else:
                    raise RuntimeError(
                        f"No image data in response item {i}"
                    )

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": self.default_content,
                        "images": images,
                    }
                }]
            }

    # -- Wire everything together -----------------------------------------

    global _generator_factory
    _generator_factory = LumenfallImageGenerator

    from .provider import LumenfallProvider

    ctx.add_provider(LumenfallProvider)
    ctx.add_provider(LumenfallImageGenerator)


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
