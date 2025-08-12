# red-gpt5assistant

A production-ready Red-DiscordBot cog that replaces aiuser with a new OpenAI Responses API implementation, using GPT‑5 and native tools (Web Search, File Search, Code Interpreter), plus `gpt-image-1` for image generation and edits.

## Features

- GPT-5 chat via Responses API with built-in tools
- Per-guild config: model, verbosity, reasoning effort, tools, system prompt
- Streaming to Discord with typing indicator and safe 2,000-char chunking (code-fence aware)
- Image generation + edit with user attachments
- File uploads to a per-guild knowledge base (File Search)
- Prefix commands (`[p]gpt5 ...`); can be used alongside other cogs
- Retry/backoff on transient OpenAI errors

## Quickstart (5 minutes)

1. Install the repo into Red’s downloader:
   - `[p]repo add aiuserOAI-Codex https://github.com/mjnhbg3/aiuserOAI-Codex`
   - `[p]cog install aiuserOAI-Codex gpt5assistant`
   - `[p]load gpt5assistant`
2. Set your OpenAI API key (requires OpenAI SDK >= 1.99.0; this repo specifies it and Red installs it automatically on first install):
   - `[p]set api openai api_key,<YOUR_KEY>`
3. Configure tools and model (optional):
   - `[p]gpt5 config model gpt-5`
   - `[p]gpt5 config tools enable web_search`
   - `[p]gpt5 config tools enable file_search`
4. Ask something:
   - `[p]gpt5 ask what’s new with the James Webb telescope?`
5. Generate an image:
   - `[p]gpt5 image a flat, minimal owl logo`
   - For edits: attach an image and run `[p]gpt5 image add a neon outline`
6. Upload files for File Search:
   - Attach PDFs and run `[p]gpt5 upload`

## Commands

- `[p]gpt5 status` — Show current config and enabled tools
- `[p]gpt5 ask <text>` — Force chat path
- `[p]gpt5 image <prompt>` — Generate (or edit if an image is attached)
- `[p]gpt5 upload` — Upload attached files to the guild knowledge base
- `[p]gpt5 config model <name>`
- `[p]gpt5 config reasoning <minimal|low|medium|high>`
  - Maps to Responses API control: `reasoning.effort`.
- `[p]gpt5 config tools <enable|disable> <web_search|file_search|code_interpreter|image>`
- `[p]gpt5 config channels <allow|deny> [#channel]`
- `[p]gpt5 config system <prompt>`
- `[p]gpt5 config max_tokens <n>`
- `[p]gpt5 config temperature <0..2>`
- `[p]gpt5 config privacy <ephemeral|off>`

## Behavior Notes

- If web search is disabled, GPT-5 answers without live data.
- File Search: after uploading, the cog attaches file IDs to chat requests so GPT-5 can retrieve context.
- Code Interpreter: when enabled, the model may run Python to solve tasks; artifacts are summarized in output.
- Safety: No `@everyone` mentions, NSFW and Discord limits respected.

## Costs & Privacy

- The cog sends user messages, system prompts, and optionally uploaded files to OpenAI for processing.
- Disable tools you do not want to use with `[p]gpt5 config tools disable ...`.

## Migration from aiuser

- Replaces custom tool shims (Serper/scrape) with native `web_search`.
- `model`, `verbosity`, `reasoning`, and `tools` map directly to GPT‑5 controls.
- Per-guild allowlist and system prompt preserved.

## Development

- Python 3.11+
- `pytest`, `ruff`, `mypy` configured via `pyproject.toml`.
- Tests mock OpenAI client; no network required.

## License

MIT
