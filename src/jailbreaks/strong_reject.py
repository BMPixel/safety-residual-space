# extract ground_truth

import pandas as pd
from stealthy_toxicity.guard import pred_moderation

from src.strong_reject.load_datasets import load_strongreject

ds = load_strongreject()
df = ds.to_pandas()

df = pred_moderation(df, target_input_col="forbidden_prompt")

df.to_csv("data/sr_moderation.csv")
