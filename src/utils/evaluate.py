import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.chat import eval_moderation, pred_moderation
from src.utils.dpo_dataset import make_dataset_dpo

# debug
with open("logs/warning.log", "w") as f:
    f.truncate(0)
import warnings


def custom_warning(message, category, filename, lineno, file=None, line=None):
    with open("logs/warning.log", "a") as f:
        f.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = custom_warning
import sys
import traceback

from IPython import embed


def excepthook(type, value, tb):
    traceback.print_exception(type, value, tb)
    embed()


def main():
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Zero-shot model ID"
    )
    parser.add_argument("-d", "--dir", default=None, help="Checkpoint directory (optional)")
    args = parser.parse_args()

    zero_shot_model = args.model
    dir = args.dir
    TAG = os.path.basename(dir) if dir else os.path.basename(zero_shot_model)
    result_dir = "artifacts"

    ckpts = [zero_shot_model]
    if dir:
        ckpts += [
            f"{dir}/t10_n1",
            f"{dir}/t20_n1",
            f"{dir}/t40_n1",
            f"{dir}/t80_n1",
            f"{dir}/t160_n1",
        ]

    # pred
    results = {}
    test_set = make_dataset_dpo(160, 60)["test"].to_pandas()

    sample_indices = list(range(0, len(test_set), len(test_set) // 20))[:20]
    for ckpt in ckpts:
        os.environ["DISABLE_QUERY_CACHE"] = "1"  # disable cache
        # if 'DISABLE_QUERY_CACHE' in os.environ: # enable cache
        #     del os.environ['DISABLE_QUERY_CACHE']
        test_df = pred_moderation(test_set, target_input_col="prompt", model_id=ckpt)
        if "DISABLE_QUERY_CACHE" in os.environ:
            del os.environ["DISABLE_QUERY_CACHE"]

        results[ckpt] = eval_moderation(test_df, category_col="method")

        print(f"\n{'='*10} Checkpoint: {ckpt.split('/')[-1]} {'='*10}")
        print("\nSample outputs:")
        for idx in sample_indices:
            row = test_df.iloc[idx]
            print(f"\n[{idx}] {row['method']} ({row['tag']}) - Score: {row['score']:.3f}")
            print(f"Prompt: {row['prompt']}")
            print(f"Response: {row['moderation_pred']}")
        print(f"\nMetrics: {json.dumps(results[ckpt], indent=2)}\n")

    # Only create exposure analysis if directory is provided
    if dir:
        scores_by_exposure = {
            "orig": {k: v for k, v in results[zero_shot_model].items() if "avg_score" in k},
            "10": {k: v for k, v in results[f"{dir}/t10_n1"].items() if "avg_score" in k},
            "20": {k: v for k, v in results[f"{dir}/t20_n1"].items() if "avg_score" in k},
            "40": {k: v for k, v in results[f"{dir}/t40_n1"].items() if "avg_score" in k},
            "80": {k: v for k, v in results[f"{dir}/t80_n1"].items() if "avg_score" in k},
            "160": {k: v for k, v in results[f"{dir}/t160_n1"].items() if "avg_score" in k},
        }

        scores_df = pd.DataFrame(scores_by_exposure).transpose()
        scores_df.to_csv(f"{result_dir}/{TAG}.csv", index_label="t")

        plt.figure(figsize=(10, 6), dpi=300)
        x = range(len(scores_df.index))
        for column in scores_df.columns:
            plt.plot(x, scores_df[column], marker="o", label=column)
        plt.xticks(x, scores_df.index, rotation=45)
        plt.xlabel("Exposure")
        plt.ylabel("Average Score")
        plt.title(f"Average Scores by Exposure: {TAG}")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{result_dir}/{TAG}.png")
        plt.close()
    else:
        print(
            f"\nZero-shot evaluation complete. Results: {json.dumps(results[zero_shot_model], indent=2)}"
        )


def with_exception():
    sys.excepthook = excepthook
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        embed()


if __name__ == "__main__":
    main()
