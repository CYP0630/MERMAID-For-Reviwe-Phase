from __future__ import annotations

import os
import json
import logging
import colorlog

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv
from together import Together
from openai import OpenAI, AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log


load_dotenv()

# ---------------------------------------------------------------------------
#   Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   OpenAI backend
# ---------------------------------------------------------------------------
class ChatBackend:
    def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIBackend(ChatBackend):
    def __init__(self, 
                model: str,
                key: str,
                url: str):
        self.model = model
        self.client = OpenAI(
            api_key=key,
            base_url=url,
        ) 
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 15000,
    ) -> Dict[str, Any]:
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            
        resp = self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        msg = resp.choices[0].message
        
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        
        if raw_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}


class TogetherAIBackend(ChatBackend):
    def __init__(self, 
                model: str,
                key: str,
                url: str):
        self.model = model
        self.client = Together(
            api_key=key,
            base_url=url,
        ) 
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            
        resp = self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        msg = resp.choices[0].message
        
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        if raw_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}
