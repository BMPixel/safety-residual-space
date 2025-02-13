import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.extract import calculate_aligned_diffs, calculate_components, get_activations
from src.utils.ppl_checker import calculate_perplexity
from src.utils.ssft_dataset import make_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run perplexity evaluation with intervention")
    parser.add_argument("--base_model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ft_model_path", default="ckpts/dpo/t160_n1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer_idx", type=int, default=25)
    parser.add_argument("--component_idx", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--max_dalpha", type=float, default=None)
    parser.add_argument("--run_type", choices=["base", "ft", "intervention"], required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    data = make_dataset(50, 0, 700, 256)["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    chat_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    if args.run_type == "base":
        model = AutoModelForCausalLM.from_pretrained(args.base_model_path, device_map=args.device)
        ppl = calculate_perplexity(model, chat_tokenizer)
        print(f"Base model perplexity: {ppl:.5f}")
    elif args.run_type == "ft":
        model = AutoModelForCausalLM.from_pretrained(args.ft_model_path, device_map=args.device)
        ppl = calculate_perplexity(model, chat_tokenizer)
        print(f"Finetuned model perplexity: {ppl:.5f}")
    elif args.run_type == "intervention":
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
        ft_model = AutoModelForCausalLM.from_pretrained(args.ft_model_path)

        base_activations, _ = get_activations(base_model, tokenizer, data["input"])
        ft_activations, _ = get_activations(ft_model, tokenizer, data["input"])
        _, _, diff_activations = calculate_aligned_diffs(
            base_activations, ft_activations, method="matrix"
        )

        diff_components_by_layer = []
        for layer in range(diff_activations.shape[1]):
            comps, _ = calculate_components(diff_activations[:, layer, :])
            diff_components_by_layer.append(comps)

        direction = diff_components_by_layer[args.layer_idx][args.component_idx]
        model = AutoModelForCausalLM.from_pretrained(args.ft_model_path, device_map=args.device)
        ppl = calculate_perplexity(
            model,
            chat_tokenizer,
            direction=direction,
            alpha=args.alpha,
            use_proj=args.use_proj,
            max_dalpha=args.max_dalpha,
        )
        print(f"Intervention perplexity: {ppl:.5f}")


if __name__ == "__main__":
    main()

# Reproduce original results with these commands:
# Base model:          python perplexity_evaluation.py --run_type base --device cuda:5
# Finetuned model:     python perplexity_evaluation.py --run_type ft --device cuda:4
# Non-dominant sup:    python perplexity_evaluation.py --run_type intervention --layer_idx 14 --component_idx 2:6 --alpha 4.5 --use_proj --max_dalpha 4
# Dominant direction:  python perplexity_evaluation.py --run_type intervention --layer_idx 25 --component_idx 0 --alpha -1
# PAIR intervention:   python perplexity_evaluation.py --run_type intervention --layer_idx 15 --component_idx 6 --alpha 1.15
