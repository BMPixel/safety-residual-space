import pandas as pd

from src.strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from src.strong_reject.load_datasets import load_strongreject

forbidden_prompt_dataset = load_strongreject()
jailbroken_dataset = apply_jailbreaks_to_dataset(forbidden_prompt_dataset, ["pair"], verbose=True)

jailbroken_dataset.to_pandas().to_csv("data/pair_prompts.csv")
