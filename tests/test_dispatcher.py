import sys
import types
import asyncio
import pytest
from dataclasses import dataclass


# Stub minimal redbot.core and discord before imports
mod_red = types.ModuleType("redbot")
mod_core = types.ModuleType("redbot.core")
mod_commands = types.ModuleType("redbot.core.commands")

class FakeConfig:
    def __init__(self, data):
        self._data = data

    def guild(self, _):
        return self

    async def all(self):
        return self._data


mod_core.Config = FakeConfig  # type: ignore
mod_commands.Bot = object  # type: ignore
mod_commands.Cog = object  # type: ignore
mod_commands.Context = object  # type: ignore
mod_core.commands = mod_commands
mod_red.core = mod_core
sys.modules.setdefault("redbot", mod_red)
sys.modules.setdefault("redbot.core", mod_core)
sys.modules.setdefault("redbot.core.commands", mod_commands)

# Stub gpt5assistant.openai_client before importing Dispatcher
mod_oai = types.ModuleType("gpt5assistant.openai_client")

@dataclass
class ChatOptions:
    model: str
    tools: dict
    reasoning: str
    max_tokens: int
    temperature: float
    system_prompt: str
    file_ids: list | None
    vector_store_id: str | None = None
    inline_file_ids: list | None = None
    inline_image_ids: list | None = None


class OpenAIClient:  # not used here
    pass


mod_oai.ChatOptions = ChatOptions  # type: ignore
mod_oai.OpenAIClient = OpenAIClient  # type: ignore
sys.modules.setdefault("gpt5assistant.openai_client", mod_oai)

# Minimal discord stub
mod_discord = types.ModuleType("discord")
sys.modules.setdefault("discord", mod_discord)

from gpt5assistant.dispatcher import Dispatcher


class DummyAuthor:
    def __init__(self):
        self.bot = False


class DummyChannel:
    def __init__(self):
        self.sent = []

    async def send(self, content=None, **kwargs):
        self.sent.append(content)
        class Msg:
            pass
        return Msg()

    def typing(self):
        class _T:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _T()


class DummyGuild:
    id = 123


class DummyMessage:
    def __init__(self, content):
        self.content = content
        self.guild = DummyGuild()
        self.channel = DummyChannel()
        self.author = DummyAuthor()
        self.attachments = []
        self.mentions = []


class FakeClient:
    async def respond_chat(self, msgs, options):
        async def gen():
            for tok in ["Hello ", "from ", "GPT5!"]:
                await asyncio.sleep(0)
                yield tok
        return gen()


@pytest.mark.asyncio
async def test_dispatcher_chat_streams_chunks():
    cfg = {
        "model": "gpt-5",
        "verbosity": "low",
        "reasoning": "minimal",
        "tools": {"web_search": False, "file_search": False, "code_interpreter": False, "image": True},
        "allowed_channels": [],
        "system_prompt": "",
        "max_tokens": 100,
        "temperature": 0.7,
        "ephemeral": False,
        "file_ids": [],
        "respond_on_mention": True,
        "random_autoreply": False,
        "random_rate": 0.0,
    }
    class Bot:
        class User:
            id = 42

        user = User()

        async def get_valid_prefixes(self, _guild):
            return ["!"]

    disp = Dispatcher(bot=Bot(), config=FakeConfig(cfg), client=FakeClient())
    # Mention the bot to trigger reply
    msg = DummyMessage("<@42> say hi")
    msg.mentions = [types.SimpleNamespace(id=42)]
    await disp.handle_message(msg)
    assert "".join(msg.channel.sent) == "Hello from GPT5!"
