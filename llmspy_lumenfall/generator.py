"""
Image generator backed by Lumenfall's OpenAI-compatible API.

Supports both image generation (text-to-image) and image editing
(image+text-to-image) via Lumenfall's unified API.

Generation uses JSON POST to /images/generations.
Editing uses multipart/form-data POST to /images/edits.
"""

import base64
import json
import os
import time

import aiohttp

from llms.main import GeneratorBase


def _detect_media_type(raw):
    """Detect image MIME type from magic bytes."""
    if raw[:4] == b"\x89PNG":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _read_local_file(file_path):
    """Read a local image file from disk, return (media_type, b64_data) or None."""
    if not os.path.isfile(file_path):
        return None
    with open(file_path, "rb") as f:
        raw = f.read()
    media_type = _detect_media_type(raw)
    return (media_type, base64.b64encode(raw).decode())


def _read_cache_file(cache_url):
    """Read a /~cache/ file from disk, return (media_type, b64_data) or None."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".llms", "cache")
    relative = cache_url[len("/~cache/"):]
    file_path = os.path.join(cache_dir, relative)
    return _read_local_file(file_path)


def _extract_images_from_content(content_parts):
    """Extract (media_type, b64_data) tuples from a list of content parts.

    Handles data: URIs, /~cache/ paths, and absolute file paths.
    """
    images = []
    for part in content_parts:
        if part.get("type") != "image_url":
            continue
        url = part.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            header, _, b64 = url.partition(",")
            media_type = header.split(";")[0].replace("data:", "")
            images.append((media_type, b64))
        elif url.startswith("/~cache/"):
            result = _read_cache_file(url)
            if result:
                images.append(result)
        elif os.path.isabs(url):
            result = _read_local_file(url)
            if result:
                images.append(result)
    return images


def _model_supports_image_input(model_id):
    """Check if the model supports image input (editing) via the catalog."""
    from .models import get_models

    models = get_models()
    model_info = models.get(model_id)
    if not model_info:
        return True  # unknown model — let the API decide
    return "image" in model_info.get("modalities", {}).get("input", [])


def extract_user_images(chat):
    """Extract input images for editing from chat messages.

    Collects images from two sources and combines them:
    1. Last assistant output image (the result being iteratively edited)
    2. User-attached images from the last user message (new images from input)

    All collected images are returned together (assistant output first, then
    user-attached). If no images are found from either source, returns an
    empty list which routes to /images/generations.

    Returns (images, has_user_attached) where images is a list of
    (media_type, base64_data) tuples and has_user_attached indicates whether
    any images came from the user's message (not just assistant output).
    """
    messages = chat.get("messages", [])

    images = []
    has_user_attached = False

    # 1. Last assistant output image (for conversational editing)
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        assistant_images = msg.get("images") or []
        if not assistant_images:
            # Also check nested message dict
            inner = msg.get("message", {})
            assistant_images = inner.get("images") or []
        if assistant_images:
            extracted = _extract_images_from_content(assistant_images)
            if extracted:
                images.extend(extracted)
        break  # only check the most recent assistant message

    # 2. User-attached images from last user message
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            extracted = _extract_images_from_content(content)
            if extracted:
                images.extend(extracted)
                has_user_attached = True
        break  # only check the last user message

    return images, has_user_attached


class LumenfallImageGenerator(GeneratorBase):
    """Image generator backed by Lumenfall's OpenAI-compatible API.

    Extends GeneratorBase to implement:
    - chat() — calls Lumenfall /images/generations or /images/edits
    - to_response() — decodes images, caches to disk via
      ctx.save_image_to_cache, returns message.images with /~cache/ paths
    """

    sdk = "llmspy_lumenfall/image"

    def __init__(self, ctx, **kwargs):
        super().__init__(**kwargs)
        self.ctx = ctx
        api = kwargs.get("api", "").rstrip("/")
        # Strip endpoint suffix if present to get the base URL
        for suffix in ("/images/generations", "/images/edits"):
            if api.endswith(suffix):
                api = api[: -len(suffix)]
                break
        self.api_base = api
        self.api = self.api_base + "/images/generations"

    # -- API call -----------------------------------------------------

    async def chat(self, chat, provider=None, context=None):
        model = chat.get("model", "")
        started_at = time.time()

        prompt = self.ctx.last_user_prompt(chat)
        if not prompt:
            raise ValueError("No prompt found in chat messages")

        user_images, has_user_attached = extract_user_images(chat)
        headers = self.get_headers(provider, chat)

        if user_images and not _model_supports_image_input(model):
            if has_user_attached:
                raise ValueError(
                    f"Model '{model}' does not support image editing"
                )
            # Conversational context — fall back to generation
            user_images = []

        if user_images:
            return await self._chat_edit(
                chat, model, prompt, user_images, headers, started_at
            )
        else:
            return await self._chat_generate(
                chat, model, prompt, headers, started_at
            )

    async def _chat_generate(self, chat, model, prompt, headers, started_at):
        """Image generation — JSON POST to /images/generations."""
        aspect_ratio = self.ctx.chat_to_aspect_ratio(chat) or "1:1"

        payload = {
            "model": model,
            "prompt": prompt,
            "n": chat.get("n", 1),
            "response_format": "b64_json",
            "aspect_ratio": aspect_ratio,
        }

        self.ctx.log(f"POST {self.api}")
        self.ctx.log(json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                return await self._handle_response(response, chat, started_at)

    async def _chat_edit(
        self, chat, model, prompt, user_images, headers, started_at
    ):
        """Image editing — multipart/form-data POST to /images/edits."""
        edit_url = self.api_base + "/images/edits"

        form = aiohttp.FormData()
        form.add_field("prompt", prompt)
        form.add_field("model", model)
        form.add_field("n", str(chat.get("n", 1)))
        form.add_field("response_format", "b64_json")

        aspect_ratio = self.ctx.chat_to_aspect_ratio(chat) or "1:1"
        form.add_field("aspect_ratio", aspect_ratio)

        # Send all images as image[] fields (array notation)
        for i, (media_type, b64_data) in enumerate(user_images):
            ext = media_type.split("/")[-1] if "/" in media_type else "png"
            image_bytes = base64.b64decode(b64_data)
            form.add_field(
                "image[]",
                image_bytes,
                filename=f"image-{i}.{ext}",
                content_type=media_type,
            )

        # Remove Content-Type so aiohttp can set the multipart boundary
        headers.pop("Content-Type", None)
        headers.pop("content-type", None)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                edit_url,
                headers=headers,
                data=form,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                return await self._handle_response(response, chat, started_at)

    # -- Shared response handling -------------------------------------

    async def _handle_response(self, response, chat, started_at):
        """Shared error handling and response processing for both paths."""
        text = await response.text()
        self.ctx.log(text[:1024] + ("..." if len(text) > 1024 else ""))

        model = chat.get("model", "")

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
            raise RuntimeError(f"API error ({response.status}): {msg}")

        return self.ctx.log_json(
            await self.to_response(json.loads(text), chat, started_at)
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
            self.ctx.log(json.dumps(response, indent=2))
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
                self.ctx.log(f"GET {image_url}")
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
                relative_url, _info = self.ctx.save_image_to_cache(
                    image_bytes,
                    f"{chat.get('model', 'image')}-{i}.{ext}",
                    self.ctx.to_file_info(chat),
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
