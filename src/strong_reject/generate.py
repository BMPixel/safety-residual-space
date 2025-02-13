"""Utilities for calling LLM APIs."""

import multiprocessing
import os
import time
import warnings
from typing import Union

import diskcache as dc
import openai
from datasets import Dataset, concatenate_datasets
from diskcache import Lock

cache = dc.Cache(".cache")
lock = Lock(cache, "lock_generate")


def generate(
    prompt: Union[str, list[str]],
    model: str,
    system_prompt: str = None,
    num_retries: int = 5,
    delay: int = 1,
    **kwargs,
) -> str:
    """Call an LLM to generate a response.

    Currently supported models are:

    - Any model in Openrouter

    Args:
        prompt (Union[str, list[str]]): Prompt to respond to. If this is a list, it will be converted into a conversating with alternating "user" and "assistant" roles.
        model (str): Model.
        system_prompt (str, optional): System prompt. Defaults to None.
        num_retries (int, optional): Number of retries if an error is encountered (e.g., rate limit). Defaults to 5.
        delay (int, optional): Initial delay before calling the API. Defaults to 0.

    Returns:
        str: Response.
    """
    if num_retries == 0:
        msg = f"Failed to get response from model {model} for prompt {prompt}"
        warnings.warn(msg)
        return ""

    if delay > 0:
        time.sleep(delay)

    # construct messages
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        # prompt is a list of alternating user and assistant messages
        for i, content in enumerate(prompt):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": content})

    try:
        model_kwargs = {
            # together ai
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1",
            },
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1",
            },
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1",
            },
            "meta-llama/Meta-Llama-Guard-3-8B": {
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1",
            },
            "mistralai/Mixtral-8x22B-Instruct-v0.1": {
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1",
            },
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1",
            },
        }
        default_kwargs = {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
        }

        # Match model names to the Openrouter API
        if model.startswith("gpt"):
            model = "openai/" + model
        key = "#".join([str(x) for x in [model, messages, kwargs]])
        if not os.getenv("DISABLE_QUERY_CACHE"):
            with lock:
                if key in cache:
                    return cache[key]
        with openai.OpenAI(**model_kwargs.get(model, default_kwargs)) as client:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            if response.choices is None:
                raise Exception("No response")
            else:
                content = response.choices[0].message.content
            with lock:
                cache[key] = content
            return content
    except Exception as e:
        print(e)
        return generate(
            prompt,
            model,
            system_prompt=system_prompt,
            num_retries=num_retries - 1,
            delay=2 * max(delay, 1),
            **kwargs,
        )


def generate_to_dataset(
    dataset: Dataset,
    models: list[str],
    target_column: str = "prompt",
    decode_responses: bool = True,
    **kwargs,
) -> Dataset:
    """Generate responses to a HuggingFace dataset of prompts.

    Args:
        dataset (Dataset): Dataset with prompts.
        models (list[str]): Models used to generate responses.
        target_column (str, optional): Column that the models should use as the prompt. Defaults to "prompt".
        decode_responses (bool, optional): Decode the raw responses. Defaults to True.

    Returns:
        Dataset: Dataset with responses.
    """
    generated_datasets = []
    for model in models:
        generated_dataset = dataset.map(
            lambda x: {"response": generate(x[target_column], model, **kwargs)},
            num_proc=1,
            # num_proc=multiprocessing.cpu_count(),
            # multiprocessing_context='fork'
        )
        generated_dataset = generated_dataset.add_column("model", len(generated_dataset) * [model])
        generated_datasets.append(generated_dataset)

    dataset = concatenate_datasets(generated_datasets)
    if decode_responses:
        from .jailbreaks import decode_dataset

        dataset = decode_dataset(dataset)

    return dataset
