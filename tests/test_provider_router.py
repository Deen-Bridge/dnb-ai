import pytest

from providers.router import ProviderRouter
from providers.types import GenerationConfig, Message, ProviderReply


class FakeProvider:
    def __init__(self, name, failures=0):
        self.name = name
        self.failures = failures
        self.calls = 0

    async def generate(self, messages, *, system, config):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(self.name)
        return ProviderReply(f"reply-{self.name}", self.name)


@pytest.mark.asyncio
async def test_router_fails_over_and_preserves_order():
    primary = FakeProvider("primary", failures=1)
    fallback = FakeProvider("fallback")
    router = ProviderRouter([primary, fallback], failure_threshold=3)
    reply = await router.generate([Message("user", "hello")], system=None, config=GenerationConfig())
    assert reply.provider == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_router_opens_and_half_opens_circuit():
    primary = FakeProvider("primary", failures=3)
    fallback = FakeProvider("fallback")
    router = ProviderRouter([primary, fallback], failure_threshold=2, cooldown_seconds=0)
    await router.generate([Message("user", "one")], system=None, config=GenerationConfig())
    await router.generate([Message("user", "two")], system=None, config=GenerationConfig())
    assert router.health["primary"].state == "open"
    await router.generate([Message("user", "three")], system=None, config=GenerationConfig())
    assert primary.calls == 3
    assert router.health["primary"].state == "open"
