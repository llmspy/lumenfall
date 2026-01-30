# Lumenfall for llms.py

[Lumenfall](https://lumenfall.ai) is an AI media gatway offering unified access to the top AI image models across all major providers.

[llms.py](https://github.com/ServiceStack/llms) is a lightweight CLI and web UI to access hundreds of AI models across many providers. It's out of the box support for image models is limited.

With the Lumenfall extension, you can access all of our image models inside llms.py.

## Quick start

```bash
pip install llms-py
llms --add lumenfall-ai/llmspy-lumenfall
export LUMENFALL_API_KEY=lmnfl_your_api_key
llms --out image "A capybara relaxing in a hot spring" -m gemini-3-pro-image
```

Get your API key at [lumenfall.ai](https://lumenfall.ai) Dashboard - API Keys.

## How it works

This extension registers Lumenfall as an image generation provider inside llmspy. When you set `LUMENFALL_API_KEY`, the extension auto-registers on startup - no `llms.json` config needed.

```
llms --out image "prompt" -m <any-lumenfall-model>
```

All image requests route through Lumenfall's unified API, which handles provider routing, billing, and model access behind the scenes.

## Installation

### Via llms --add (recommended)

```bash
llms --add lumenfall-ai/llmspy-lumenfall
```

This clones the extension into `~/.llms/extensions/` and installs Python dependencies automatically.

### Via pip

```bash
pip install llmspy-lumenfall
```

### Manual (development)

```bash
git clone https://github.com/lumenfall-ai/llmspy-lumenfall.git
ln -s "$(pwd)/llmspy-lumenfall/llmspy_lumenfall" ~/.llms/extensions/llmspy_lumenfall
pip install -r llmspy-lumenfall/requirements.txt
```

## Configuration

Set your API key as an environment variable:

```bash
export LUMENFALL_API_KEY=lmnfl_your_api_key
```

That's it. The extension auto-registers when it detects the key. No `llms.json` entry required.

### Optional: explicit provider config

If you prefer explicit configuration, add this to `~/.llms/llms.json`:

```json
{
  "providers": {
    "lumenfall": { "enabled": true }
  }
}
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LUMENFALL_API_KEY` | Yes | Your Lumenfall API key (starts with `lmnfl_`) |
| `LUMENFALL_BASE_URL` | No | Override API base URL (default: `https://api.lumenfall.ai/openai/v1`) |

## Usage

### Generate a single image

```bash
llms --out image "A capybara wearing a tiny hat" -m gemini-3-pro-image
```

### Generate multiple images

```bash
llms --out image "A capybara swimming in a hot spring" -m gpt-image-1.5 -n 2
```

### Get raw JSON response

```bash
llms --out image "A capybara riding a skateboard" -m flux.2-max --raw
```

### Set aspect ratio

Add to `~/.llms/llms.json`:

```json
{
  "defaults": {
    "out:image": {
      "image_config": { "aspect_ratio": "16:9" }
    }
  }
}
```

Supported ratios: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `21:9`

## Available models

Lumenfall strives to offer every image model that exists, backend by multiple providers.
You can find our current selection in the [Lumenfall model catalog](https://lumenfall.ai/models).

For models that are available on other providers in llmspy, the first match is used. See "Routing" below to control this.

## Routing: avoiding model conflicts

If you also use Google or OpenAI providers for text, restrict them to text models if you want all image models to be served by Lumenfall:

```json
{
  "providers": {
    "google": {
      "enabled": true,
      "map_models": {
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro"
      }
    },
    "openai": {
      "enabled": true,
      "map_models": {
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "o4-mini": "o4-mini"
      }
    },
    "lumenfall": { "enabled": true }
  }
}
```

`map_models` whitelists which models a provider serves. By listing only text models for Google/OpenAI, all image models route through Lumenfall.

### Project structure

```
llmspy_lumenfall/
  __init__.py    # Extension entry point (install/load hooks, generator)
  provider.py    # LumenfallProvider extending OpenAiCompatible
  models.py      # Model catalog
tests/
  conftest.py    # Test fixtures and helpers
  test_e2e.py    # 10 end-to-end tests
```

## Links
- [Lumenfall website (Create an API key)](https://lumenfall.ai)
- [Lumenfall llmspy documentation](https://docs.lumenfall.ai/integrations/llmspy)
- [Model catalog](https://lumenfall.ai/models)
- [Lumenfall documentation](https://docs.lumenfall.ai)
