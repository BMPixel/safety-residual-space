import os

import diskcache as dc
import pandas as pd
import torch
from datasets import Dataset
from diskcache import Lock
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.strong_reject.evaluate import evaluate, evaluate_dataset

cache = dc.Cache("query_cache")
lock = Lock(cache, "lock_generate")

tqdm.pandas()

# Global variables for lazy loading
tokenizer = None
model = None
current_model_id = None
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def get_model(model_id=MODEL_ID):
    global tokenizer, model, current_model_id
    if isinstance(model_id, str):
        if tokenizer is None or model is None or current_model_id != model_id:
            if current_model_id is not None:
                print(f"Unloading model {current_model_id}")
                del tokenizer, model
            device = "cuda"
            dtype = torch.bfloat16
            print(f"Loading model {model_id} on {device} with dtype {dtype}")

            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto"
            )
            current_model_id = model_id
        return tokenizer, model
    else:
        # Assume model_id is actually a model instance
        return model_id.tokenizer, model_id


def moderate(chat, model_id=MODEL_ID):
    tokenizer, model = get_model(model_id)
    input = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to("cuda")
    torch.manual_seed(42)
    output = model.generate(input, max_new_tokens=512, pad_token_id=0, do_sample=False)
    prompt_len = input.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def moderate_input(line: str, model_id=MODEL_ID):
    cache_key = f"moderate_input#{model_id}#{line}"

    if not os.getenv("DISABLE_QUERY_CACHE"):
        with lock:
            if cache_key in cache:
                return cache[cache_key]

    line = moderate([{"role": "user", "content": line}], model_id)
    with lock:
        cache[cache_key] = line.strip()
    return line.strip()


def moderate_batch(chats, model_id=MODEL_ID, batch_size=64):
    tokenizer, model = get_model(model_id)
    tokenizer.pad_token_id = 0
    results = []

    for i in tqdm(range(0, len(chats), batch_size), desc="processing batch"):
        batch = chats[i : i + batch_size]
        inputs = tokenizer.apply_chat_template(
            batch, tokenize=True, return_tensors="pt", padding=True
        ).to("cuda")
        torch.manual_seed(42)
        outputs = model.generate(
            inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id, do_sample=False
        )

        prompt_len = inputs.shape[1]
        for output in outputs:
            result = tokenizer.decode(output[prompt_len + 3 :], skip_special_tokens=True)
            results.append(result)
    return results


def moderate_input_batch(lines: list, model_id=MODEL_ID):
    cached_results = {}
    lines_to_process = []
    line_indices = []

    if not os.getenv("DISABLE_QUERY_CACHE"):
        with lock:
            for i, line in enumerate(lines):
                cache_key = f"moderate_input#{model_id}#{line}"
                if cache_key in cache:
                    cached_results[i] = cache[cache_key]
                else:
                    lines_to_process.append(line)
                    line_indices.append(i)
    else:
        lines_to_process = lines
        line_indices = list(range(len(lines)))

    if lines_to_process:
        chats = [[{"role": "user", "content": line}] for line in lines_to_process]
        results = moderate_batch(chats, model_id)

        with lock:
            for idx, result in zip(line_indices, results):
                cache_key = f"moderate_input#{model_id}#{lines[idx]}"
                cache[cache_key] = result.strip()
                cached_results[idx] = result.strip()
    return [cached_results[i] for i in range(len(lines))]


def pred_moderation(df, model_id=MODEL_ID, target_input_col="prompt"):
    df = df.copy()
    # df['moderation_pred'] = df[target_input_col].progress_apply(lambda x: moderate_input(x, model_id))

    prompts = df[target_input_col].tolist()
    results = moderate_input_batch(prompts, model_id)
    df["moderation_pred"] = [r.rstrip("!") for r in results]
    return df


# evaluate
def evaluate_item(
    data,
    evaluator: str = "pair",  # pair or pair_safe
    models: list[str] = ["gpt-4o-mini", "gpt-4o", "nousresearch/hermes-3-llama-3.1-70b"],
):
    if data["moderation_pred"] == "":
        print(f"Empty response for {data['prompt']}")
        return 0.0
    prompt = data["prompt"]
    response = data["moderation_pred"]
    score = evaluate(
        forbidden_prompt=prompt, response=response, evaluators=[evaluator], models=models
    )

    return score[0]["score"]


def eval_moderation(
    df,
    category_col="method",
    evaluator: str = "strongreject_rubric",  # pair or pair_safe
    models: list[str] = [
        "gpt-4o-mini",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "nousresearch/hermes-3-llama-3.1-70b",
        "gpt-4o",
    ],
):
    # get score
    generated_dataset = Dataset.from_pandas(df)
    evaluated_dataset = evaluate_dataset(
        generated_dataset,
        [evaluator],
        forbidden_prompt_column="forbidden_prompt",
        response_column="moderation_pred",
        models=models,
    )
    df["score"] = evaluated_dataset["score"]

    # get accuracy
    metrics = {}
    for method in set(df[category_col]):
        method_df = df[df[category_col] == method]

        if len(method_df[method_df["tag"] == "toxic"]) > 0:
            avg_score_toxic = method_df[method_df["tag"] == "toxic"]["score"].mean()
            metrics[f"{method}_toxic_avg_score"] = float(avg_score_toxic)
            print(f"method: {method}, len: {len(method_df[method_df["tag"] == "toxic"])}")
        if len(method_df[method_df["tag"] == "safe"]) > 0:
            avg_score_safe = method_df[method_df["tag"] == "safe"]["score"].mean()
            metrics[f"{method}_safe_avg_score"] = float(avg_score_safe)
            print(f"method: {method}, len: {len(method_df[method_df["tag"] == "safe"])}")

    return metrics
