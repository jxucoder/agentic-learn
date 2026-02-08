"""LLM client for the MLE-bench agent.

Supports Anthropic and OpenAI-compatible providers (including DeepSeek).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class LLMResponse:
    """Response from an LLM completion."""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    """Unified LLM client for code generation and reflection.

    Supports:
    - Anthropic (Claude)
    - OpenAI (GPT, o-series)
    - OpenAI-compatible (DeepSeek, etc.) via base_url
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package required: pip install anthropic")
            self.model = model or "claude-sonnet-4-20250514"
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.client = anthropic.Anthropic(api_key=key)

        elif self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package required: pip install openai")
            self.model = model or "gpt-4o"
            key = api_key or os.environ.get("OPENAI_API_KEY")
            kwargs: dict[str, Any] = {"api_key": key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"LLM client: provider={self.provider}, model={self.model}")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            prompt: User message / prompt.
            system: System prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with the generated content.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "anthropic":
            return self._complete_anthropic(prompt, system, temp, tokens)
        else:
            return self._complete_openai(prompt, system, temp, tokens)

    def _complete_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Complete using Anthropic API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _complete_openai(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Complete using OpenAI-compatible API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Some models (o1, o3) don't support temperature
        if not self.model.startswith("o1") and not self.model.startswith("o3"):
            kwargs["temperature"] = temperature

        response = self.client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model or self.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def generate_code(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """Generate code and extract it from the response.

        Calls complete() and extracts the Python code block from the response.
        """
        response = self.complete(prompt, system=system)
        code = extract_code(response.content)

        logger.info(
            f"Generated {len(code)} chars of code "
            f"(tokens: {response.input_tokens}in/{response.output_tokens}out)"
        )

        return code


def extract_code(text: str) -> str:
    """Extract Python code from an LLM response.

    Handles:
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    - Raw code (if no blocks found)
    """
    # Try to find ```python blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Return the longest match (most complete solution)
        return max(matches, key=len).strip()

    # Try generic code blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Fallback: return everything that looks like Python code
    # (starts with import, def, class, #, or common Python patterns)
    lines = text.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "#!", '"""', "'''")) or in_code:
            in_code = True
            code_lines.append(line)
        elif in_code and (stripped == "" or stripped.startswith("#") or not stripped.startswith(("*", "-", ">"))):
            code_lines.append(line)
        elif in_code:
            # Hit non-code content, stop
            break

    if code_lines:
        return "\n".join(code_lines).strip()

    # Last resort: return the entire text
    return text.strip()
