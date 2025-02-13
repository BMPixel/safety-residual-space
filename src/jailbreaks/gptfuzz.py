import pandas as pd

from src.strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from src.strong_reject.load_datasets import load_strongreject

insert_marker = "[INSERT PROMPT HERE]"

template_df = pd.read_csv("data/gptfuzz_templates.csv")


@register_jailbreak("gptfuzz")
def gptfuzz(prompt, **kwargs):
    idx = len(prompt) % len(template_df)
    return template_df["text"][idx].replace(insert_marker, prompt)


forbidden_prompt_dataset = load_strongreject()
jailbroken_dataset = apply_jailbreaks_to_dataset(
    forbidden_prompt_dataset, ["gptfuzz"], verbose=True
)

jailbroken_dataset.to_pandas().to_csv("data/gptfuzz_prompts.csv")
