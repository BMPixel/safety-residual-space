"""
GCG's code is in the repo https://github.com/GraySwanAI/nanoGCG/tree/main
"""

import nanogcg
import torch
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.strong_reject.generate import generate_to_dataset
from src.strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from src.strong_reject.load_datasets import load_strongreject


# GCG
@register_jailbreak("gcg")
def gcg(prompt, model, tokenizer, target="Sure, here's"):
    config = GCGConfig(num_steps=500, search_width=64, topk=64, seed=42, verbosity="WARNING")
    result = nanogcg.run(
        model=model, tokenizer=tokenizer, messages=prompt, target=target, config=config
    )
    return prompt + result.best_string


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    forbidden_prompt_dataset = load_strongreject()
    jailbroken_dataset = apply_jailbreaks_to_dataset(
        forbidden_prompt_dataset, ["gcg"], model=model, tokenizer=tokenizer
    )
    jailbroken_dataset = generate_to_dataset(
        jailbroken_dataset,
        ["meta-llama/llama-3.1-8b-instruct"],
        target_column="jailbroken_prompt",
        decode_responses=False,
    )
    jailbroken_dataset.to_pandas().to_csv("helper/baseline/data/gcg_prompts.csv")
