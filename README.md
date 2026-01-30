# Lumenfall for llms.py

[Lumenfall](https://lumenfall.ai) is an AI media gateway offering unified access to the top AI image models across all major providers.

[llms.py](https://github.com/ServiceStack/llms) is a lightweight CLI and web UI to access hundreds of AI models across many providers. Its out of the box support for image models is limited.

With the Lumenfall extension, you can access all of our image models inside llms.py.

## Quick start

```bash
llms --add lumenfall-ai/llmspy-lumenfall
export LUMENFALL_API_KEY=lmnfl_your_api_key
```

Add the provider to `~/.llms/llms.json`:

```json
{
  "providers": {
    "lumenfall": { "enabled": true, "npm": "llmspy_lumenfall" }
  }
}
```

Then generate an image:

```bash
llms --out image "A capybara relaxing in a hot spring" -m gemini-3-pro-image
```

Get your API key at [lumenfall.ai](https://lumenfall.ai) Dashboard - API Keys.

## How it works

This extension registers Lumenfall as an image generation provider inside llmspy. You need to add it to your `~/.llms/llms.json` providers and set the `LUMENFALL_API_KEY` environment variable.

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

### Manual (development)

```bash
git clone https://github.com/lumenfall-ai/llmspy-lumenfall.git
ln -s "$(pwd)/llmspy-lumenfall/llmspy_lumenfall" ~/.llms/extensions/llmspy_lumenfall
pip install -r llmspy-lumenfall/requirements.txt
```

## Configuration

### 1. Set your API key

The API key is configured via the `LUMENFALL_API_KEY` environment variable. To make it available in every shell session, add it to your shell profile:

```bash
# Bash (~/.bashrc or ~/.bash_profile)
echo 'export LUMENFALL_API_KEY="lmnfl_your_api_key"' >> ~/.bashrc
source ~/.bashrc

# Zsh (~/.zshrc)
echo 'export LUMENFALL_API_KEY="lmnfl_your_api_key"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Register the provider

Add Lumenfall to the `providers` section of `~/.llms/llms.json`:

```json
{
  "providers": {
    "lumenfall": { "enabled": true, "npm": "llmspy_lumenfall" }
  }
}
```

This step is required for the extension to work.

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

## Edit images

Image editing is also supported through Lumenfall.

```bash
llms "Add a tiny sombrero to the capybara" -m gpt-image-1 --image photo.png
```

Even image editing models that usually can't be used in a conversational interface, like Seedream-4.5, can be used with turn based editing.

## Web UI

llms.py includes a built-in web UI where all Lumenfall functionality - image generation and editing - works the same as on the CLI. Start it with:

```bash
llms --server 8000
```

Then open `http://localhost:8000` in your browser.

## Available models

Lumenfall strives to offer every image model that exists, backend by multiple providers.
You can find our current selection in the [Lumenfall model catalog](https://lumenfall.ai/models).

For models that are available on other providers in llmspy, the first match is used. See "Routing" below to control this.

## Routing: avoiding model conflicts

llms.py uses the first matching provider for a given model. To ensure all image requests go through Lumenfall, list it **first** in your `providers` config - before any other providers that may also serve image models (e.g. Google, OpenAI, or OpenRouter):

```json
{
  "providers": {
    "lumenfall": { "enabled": true, "npm": "llmspy_lumenfall" },
    "google": { ... },
    "openai": { ... },
    ...
  }
}
```

## Known limitations

- **Provider forcing not supported.** Lumenfall providers cannot be forced by prepending the provider in the model (e.g. `replicate/gemini-3-pro-image`) as usual.

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
