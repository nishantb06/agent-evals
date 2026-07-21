"""Model profiles for A/B agent runs (frontier Gemini vs OSS Llama on NVIDIA).

Every agent-originated gateway chat call should use get_chat_kwargs() so
agent_routing.yaml pins and hardcoded provider=\"g\" cannot override the
chosen profile.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    model: str | None  # None → gateway default for that provider
    label: str


PROFILES: dict[str, ModelProfile] = {
    "gemini": ModelProfile(
        name="gemini",
        provider="gemini",
        model=None,
        label="gemini (frontier)",
    ),
    "llama-3": ModelProfile(
        name="llama-3",
        provider="nvidia",
        model="meta/llama-3.1-70b-instruct",
        label="llama-3 (nvidia / meta/llama-3.1-70b-instruct)",
    ),
}

_ALIASES = {
    "gemini": "gemini",
    "g": "gemini",
    "llama-3": "llama-3",
    "llama3": "llama-3",
    "llama": "llama-3",
}

DEFAULT_PROFILE = "gemini"

_current: ContextVar[ModelProfile] = ContextVar(
    "agent_model_profile",
    default=PROFILES[DEFAULT_PROFILE],
)


class UnknownModelProfile(ValueError):
    """Raised when --model / model_profile is not a known profile name."""


def resolve(name: str | None) -> ModelProfile:
    """Resolve a profile name (with aliases). None → default gemini."""
    if name is None or str(name).strip() == "":
        return PROFILES[DEFAULT_PROFILE]
    key = str(name).strip().lower()
    canon = _ALIASES.get(key)
    if canon is None or canon not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise UnknownModelProfile(
            f"unknown model profile {name!r}; choose one of: {known}"
        )
    return PROFILES[canon]


def set_profile(name: str | None) -> ModelProfile:
    """Set the process/context profile; returns the resolved profile."""
    profile = resolve(name)
    _current.set(profile)
    return profile


def get_profile() -> ModelProfile:
    return _current.get()


def get_chat_kwargs() -> dict:
    """Kwargs to merge into LLM().chat(...).

    Always sets provider so agent_routing.yaml cannot win.
    Omits model when None so the gateway uses its provider default.
    """
    p = get_profile()
    out: dict = {"provider": p.provider}
    if p.model:
        out["model"] = p.model
    return out
