"""Shared utilities for the X30 VLN framework: VLM client, image encoding, JSON repair, model config."""
from .config import ModelsConfig, VLMConfig, find_config_dir, load_models_config, load_yaml, stable_hash, file_hash
from .images import array_from_image_msg, encode_image, estimate_image_tokens, image_part, strip_images, to_data_url, user_message
from .jsonrepair import extract_text_tool_calls, loads_lenient, parse_arguments, split_think
from .toolschema import coerce_arguments, make_tool
from .vlm_client import (ChatResult, ToolCall, UsageStats, VLMCancelled, VLMClient, VLMConnectionError, VLMContextOverflow, VLMError,
                         VLMHTTPError, VLMTimeout)

__all__ = [
    "ChatResult", "ModelsConfig", "ToolCall", "UsageStats", "VLMCancelled", "VLMClient", "VLMConfig", "VLMConnectionError",
    "VLMContextOverflow", "VLMError", "VLMHTTPError", "VLMTimeout", "array_from_image_msg", "coerce_arguments", "encode_image",
    "estimate_image_tokens", "extract_text_tool_calls", "file_hash", "find_config_dir", "image_part", "load_models_config", "load_yaml",
    "loads_lenient", "make_tool", "parse_arguments", "split_think", "stable_hash", "strip_images", "to_data_url", "user_message",
]
__version__ = "0.1.0"
