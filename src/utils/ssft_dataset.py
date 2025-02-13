import os
from typing import Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from src.strong_reject.load_datasets import load_strongreject

STRONG_REJECT_NUM = 313
SAFE_OUTPUT_TEMPLATE = "\n\nsafe"
UNSAFE_OUTPUT_TEMPLATE = "\n\nunsafe\nS{N}"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load strong reject moderation truth
strong_reject_moderation_truth = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data", "sr_moderation.csv"), index_col=0
)
strong_reject_moderation_truth["type"] = "strong_reject"
adv_moderation_truth = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data", "adv_moderation.csv"), index_col=0
)
adv_moderation_truth["type"] = "adv"
or_bench_dataset = load_dataset("bench-llm/or-bench", "or-bench-80k")["train"].to_pandas()
or_bench_dataset.rename(columns={"prompt": "forbidden_prompt"}, inplace=True)
or_bench_dataset["type"] = "or_bench"

# Load harmful prompts
harmful_prompts = {
    "ours": pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "trigger_removal_prompts.csv"), index_col=0
    ),
    "flip": pd.read_csv(os.path.join(PROJECT_ROOT, "data", "flip_prompts.csv"), index_col=0),
    "pair": pd.read_csv(os.path.join(PROJECT_ROOT, "data", "pair_prompts.csv"), index_col=0),
    "gptfuzz": pd.read_csv(os.path.join(PROJECT_ROOT, "data", "gptfuzz_prompts.csv"), index_col=0),
    "simple": pd.read_csv(os.path.join(PROJECT_ROOT, "data", "simple_prompts.csv"), index_col=0),
}
# Add type to each harmful prompt dataset
for key, df in harmful_prompts.items():
    df["type"] = key

# Add moderation truth to each harmful prompt dataset
for key, df in harmful_prompts.items():
    df["moderation_truth"] = df["forbidden_prompt"].map(
        strong_reject_moderation_truth.set_index("forbidden_prompt")["moderation_truth"]
    )

# Assign output based on moderation truth
for key, df in harmful_prompts.items():
    df["output"] = df["moderation_truth"].apply(
        lambda x: UNSAFE_OUTPUT_TEMPLATE.replace("{N}", str(int(x))) if x else SAFE_OUTPUT_TEMPLATE
    )
adv_moderation_truth["output"] = adv_moderation_truth["moderation_truth"].apply(
    lambda x: UNSAFE_OUTPUT_TEMPLATE.replace("{N}", str(int(x))) if x else SAFE_OUTPUT_TEMPLATE
)
strong_reject_moderation_truth["output"] = strong_reject_moderation_truth[
    "moderation_truth"
].apply(
    lambda x: UNSAFE_OUTPUT_TEMPLATE.replace("{N}", str(int(x))) if x else SAFE_OUTPUT_TEMPLATE
)
or_bench_dataset["output"] = SAFE_OUTPUT_TEMPLATE
or_bench_dataset["moderation_truth"] = 0


def split_train_test(df: pd.DataFrame, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets."""
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42)

    # Split into train and test
    test_df = df.iloc[:test_size]
    train_df = df.iloc[test_size:]

    return train_df, test_df


def truncate_input(input: str, max_length: int) -> str:
    """Truncate the input to the max length, keep first and last half."""
    if len(input) <= max_length:
        return input

    half_length = max_length // 2
    return input[:half_length] + input[-half_length:]


def make_dataset(
    exposure_size: int, test_size_per_type: int, train_size: int = 1600, max_length: int = 2048
) -> DatasetDict:
    # Calculate training sie
    harmful_size = train_size // 2
    safe_size = train_size - harmful_size
    harmful_remaining = harmful_size - exposure_size * len(harmful_prompts)
    if harmful_remaining < 0:
        raise ValueError("Too many harmful prompts")
    elif harmful_remaining > len(strong_reject_moderation_truth) + len(adv_moderation_truth):
        raise ValueError("Not enough harmful prompts to meet training size")
    elif harmful_remaining < len(strong_reject_moderation_truth):
        strong_reject_size = harmful_remaining
        adv_size = 0
    else:
        strong_reject_size = len(strong_reject_moderation_truth)
        adv_size = harmful_remaining - strong_reject_size

    # Split training set and test set
    if exposure_size + test_size_per_type > STRONG_REJECT_NUM:
        raise ValueError(
            "Exposure size + test size per type cannot exceed the number of strong reject prompts"
        )
    harmful_prompts_train = {
        key: split_train_test(df, test_size_per_type)[0] for key, df in harmful_prompts.items()
    }
    harmful_prompts_test = {
        key: split_train_test(df, test_size_per_type)[1] for key, df in harmful_prompts.items()
    }

    # Create training dataset
    exposure_df = pd.concat(
        [
            df[["jailbroken_prompt", "output", "moderation_truth", "type"]].iloc[:exposure_size]
            for df in harmful_prompts_train.values()
        ]
    ).rename(columns={"jailbroken_prompt": "input"})
    remaining_train_df = pd.concat(
        [
            strong_reject_moderation_truth[
                ["forbidden_prompt", "output", "moderation_truth", "type"]
            ].iloc[:strong_reject_size],
            adv_moderation_truth[["forbidden_prompt", "output", "moderation_truth", "type"]].iloc[
                :adv_size
            ],
            or_bench_dataset[["forbidden_prompt", "output", "moderation_truth", "type"]].iloc[
                :safe_size
            ],
        ]
    ).rename(columns={"forbidden_prompt": "input"})
    train_df = pd.concat([exposure_df, remaining_train_df], ignore_index=True)
    if max_length is not None:
        train_df["input"] = train_df["input"].apply(lambda x: truncate_input(x, max_length))

    # Create test dataset
    exposure_df = pd.concat(
        [
            df[["jailbroken_prompt", "output", "moderation_truth", "type"]]
            for df in harmful_prompts_test.values()
        ]
    ).rename(columns={"jailbroken_prompt": "input"})
    # select last len(exposure_df) rows
    safe_df = (
        or_bench_dataset[["forbidden_prompt", "output", "moderation_truth", "type"]]
        .iloc[-len(exposure_df) :]
        .rename(columns={"forbidden_prompt": "input"})
    )
    test_df = pd.concat([exposure_df, safe_df], ignore_index=True)
    if max_length is not None:
        test_df["input"] = test_df["input"].apply(lambda x: truncate_input(x, max_length))

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        }
    )
