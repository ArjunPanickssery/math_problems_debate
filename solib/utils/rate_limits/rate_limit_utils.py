from aiolimiter import AsyncLimiter
import asyncio
import time
import tiktoken

from solib.utils import estimate_tokens, parse_time_interval

COMPLETION_MODELS = {
    "davinci-002",
    "babbage-002",
    "gpt-4-base",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-instruct-0914",
}

EMBEDDING_MODELS = (
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
)

VISION_MODELS = (
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
)

_GPT_4_MODELS = (
    "o3-mini",
    "o3-mini-2025-01-31",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1",
    "o1-2024-12-17",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
)
_GPT_3_MODELS = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
)


GPT_CHAT_MODELS = set(_GPT_4_MODELS + _GPT_3_MODELS)

OAI_FINETUNE_MODELS = (
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "davinci-002",
    "babbage-002",
)
S2S_MODELS = "gpt-4o-s2s"


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def get_max_context_length(model_id: str) -> int:
    # go to: https://platform.openai.com/docs/models
    match model_id:
        case "o1" | "o1-2024-12-17" | "o3-mini" | "o3-mini-2025-01-31":
            return 200_000
        case (
            "o1-mini"
            | "o1-mini-2024-09-12"
            | "o1-preview"
            | "o1-preview-2024-09-12"
            | "gpt-4o"
            | "gpt-4o-2024-05-13"
            | "gpt-4o-2024-08-06"
            | "gpt-4o-mini"
            | "gpt-4o-mini-2024-07-18"
            | "gpt-4-turbo"
            | "gpt-4-turbo-2024-04-09"
            | "gpt-4-0125-preview"
            | "gpt-4-turbo-preview"
            | "gpt-4-1106-preview"
            | "gpt-4-vision-preview"
        ):
            return 128_000

        case "gpt-4" | "gpt-4-0613" | "gpt-4-base":
            return 8192

        case "gpt-4-32k" | "gpt-4-32k-0613":
            return 32_768

        case "gpt-3.5-turbo-0613":
            return 4096

        case (
            "gpt-3.5-turbo"
            | "gpt-3.5-turbo-1106"
            | "gpt-3.5-turbo-0125"
            | "gpt-3.5-turbo-16k"
            | "gpt-3.5-turbo-16k-0613"
            | "babbage-002"
            | "davinci-002"
        ):
            return 16_384

        case "gpt-3.5-turbo-instruct" | "gpt-3.5-turbo-instruct-0914":
            return 4096

        case _:
            raise ValueError(f"Invalid model id: {model_id}")




def get_equivalent_model_ids(model_id: str) -> tuple[str, ...]:
    """
    Updated 2024-04-25 by Tony. This should be periodically updated.
    """

    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo#o1
    o1_models = ("o1", "o1-2024-12-17")
    if model_id in o1_models:
        return o1_models
    o1_mini_models = ("o1-mini", "o1-mini-2024-09-12")
    if model_id in o1_mini_models:
        return o1_mini_models
    o1_preview_models = ("o1-preview", "o1-preview-2024-09-12")
    if model_id in o1_preview_models:
        return o1_preview_models
    o3_mini_models = ("o3-mini", "o3-mini-2025-01-31")
    if model_id in o3_mini_models:
        return o3_mini_models

    # https://platform.openai.com/docs/models/gpt-3-5
    gpt_3_5_turbo_models = ("gpt-3.5-turbo", "gpt-3.5-turbo-0125")
    if model_id in gpt_3_5_turbo_models:
        return gpt_3_5_turbo_models
    gpt_3_5_turbo_16k_models = ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613")
    if model_id in gpt_3_5_turbo_16k_models:
        return gpt_3_5_turbo_16k_models
    gpt_3_5_turbo_instruct_models = (
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-instruct-0914",
    )
    if model_id in gpt_3_5_turbo_instruct_models:
        return gpt_3_5_turbo_instruct_models

    # https://platform.openai.com/docs/models/gpt-4o
    gpt_4o_models = ("gpt-4o", "gpt-4o-2024-08-06")
    if model_id in gpt_4o_models:
        return gpt_4o_models

    # https://platform.openai.com/docs/models/gpt-4o-mini
    gpt_4o_mini_models = ("gpt-4o-mini", "gpt-4o-mini-2024-07-18")
    if model_id in gpt_4o_mini_models:
        return gpt_4o_mini_models

    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    gpt_4_models = ("gpt-4", "gpt-4-0613")
    if model_id in gpt_4_models:
        return gpt_4_models
    gpt_4_32k_models = ("gpt-4-32k", "gpt-4-32k-0613")
    if model_id in gpt_4_32k_models:
        return gpt_4_32k_models
    gpt4t_models = ("gpt-4-turbo", "gpt-4-turbo-2024-04-09")
    if model_id in gpt4t_models:
        return gpt4t_models
    gpt4tp_models = ("gpt-4-turbo-preview", "gpt-4-0125-preview")
    if model_id in gpt4tp_models:
        return gpt4t_models

    return (model_id,)



def update_openrouter_ratelimit(model_id: str) -> tuple[int, int]:
    """https://openrouter.ai/docs/limits, returns (token_capacity, request_capacity)"""
    try:
        import requests

        url = "https://openrouter.ai/api/v1/auth/key"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

        response = requests.get(url, headers=headers)
        result = response.json()["data"][
            "rate_limit"
        ]  # {'requests': 750, 'interval': '10s'}
        num = result["requests"]
        denom = parse_time_interval(result["interval"])
        rpm = min(0.85 * 60 * num / denom, 1000)

    except Exception as e:
        import traceback

        LOGGER.error(
            f"Error getting OpenRouter rate limits:\n{e}\n{traceback.format_exc()}"
        )
        rpm = 500  # play safe until we get real rate limit

    return (200_000, rpm)



# initialize rate limits
DEFAULT_RATES = (
    {
        model: {"rpm": 4000, "tpm": 100_000}  # 4e5 but let's be even gentler
        for model in [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
    }
    | {
        "gpt-4o": {"rpm": 5000, "tpm": 450_000},
        "gpt-4o-mini": {"rpm": 5000, "tpm": 2_000_000},
        "o1-preview": {"rpm": 5000, "tpm": 450_000},
        "o1-mini": {"rpm": 5000, "tpm": 2_000_000},
        "gpt-4-turbo": {"rpm": 5000, "tpm": 450_000},
        "gpt-4": {"rpm": 5000, "tpm": 40_000},
        "gpt-3.5-turbo": {"rpm": 3500, "tpm": 2_000_000},
    }
    | {
        "gemini/gemini-1.5-pro": {"rpm": 1000, "tpm": 4e6},
        "gemini/gemini-1.5-flash": {"rpm": 2000, "tpm": 4e6},
        "gemini/gemini-1.5-flash-8b": {"rpm": 4000, "tpm": 4e6},
        "gemini/gemini-2.0-flash-exp": {
            "rpm": 10,
            "rpd": 1500,
        },  # requests per day is not enforced
    }
    | {
        "openrouter/deepseek/deepseek-chat": {"rpm": 500, "tpm": 2e5},
        "openrouter/gpt-4o-mini-2024-07-18": {"rpm": 500, "tpm": 2e5},
        "openrouter/gpt-4o-mini": {"rpm": 500},
    }
)
