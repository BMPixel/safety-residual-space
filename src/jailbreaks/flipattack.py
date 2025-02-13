import pandas as pd

from src.strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from src.strong_reject.load_datasets import load_strongreject


@register_jailbreak("flip")
def flip(prompt, **kwargs):
    return prompt[::-1]


forbidden_prompt_dataset = load_strongreject()
jailbroken_dataset = apply_jailbreaks_to_dataset(forbidden_prompt_dataset, ["flip"], verbose=True)

jailbroken_dataset.to_pandas().to_csv("data/flip_prompts.csv")
