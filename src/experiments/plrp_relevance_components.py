#!/usr/bin/env python3
import argparse
import os

import matplotlib
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("module://imgcat")
import lxt.functional as lf
from lxt.models.llama import LlamaForCausalLM, attnlrp
from transformers import AutoTokenizer

from src.utils.extract import calculate_aligned_diffs, calculate_components, get_activations
from src.utils.ssft_dataset import make_dataset

parser = argparse.ArgumentParser(description="Model Relevance Analysis")
parser.add_argument(
    "--base-model",
    type=str,
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="Path to the base model for analysis.",
)
parser.add_argument(
    "--finetuned-model",
    type=str,
    default="ckpts/ssft/t160_n1",
    help="Path to the fine-tuned model for comparison.",
)
parser.add_argument(
    "--target_layer",
    type=int,
    default=15,
    help="Index of the target layer for relevance analysis.",
)
parser.add_argument(
    "--source_layer", type=int, default=15, help="Index of the source layer for comparison."
)
parser.add_argument("--comp_num", type=int, default=5, help="Number of components to analyze.")
parser.add_argument(
    "--sample_num", type=int, default=2, help="Number of samples to use for relevance analysis."
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Device to run the model on (e.g., cuda:0, cpu)."
)

args = parser.parse_args()

BASE_MODEL = args.base_model
FINETUNED_MODEL = args.finetuned_model
DEVICE = args.device
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
output_base = f"artifacts"
os.makedirs(output_base, exist_ok=True)
data = make_dataset(50, 0, 700, 256)["train"]
samples = data["input"]
target_layer = args.target_layer
source_layer = args.source_layer
comp_num = args.comp_num
sample_num = args.sample_num

# Get indices where moderation_truth is non-zero
harmful_ids = [i for i in range(len(data["moderation_truth"])) if data["moderation_truth"][i] != 0]
safe_ids = [i for i in range(len(data["moderation_truth"])) if data["moderation_truth"][i] == 0]
pair_ids = [i for i in range(len(data["type"])) if data["type"][i] == "pair"]


def calculate_diff_components(BASE_MODEL_PATH, FINETUNED_MODEL_PATH, template="instruct"):

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    base_activations, base_logits = get_activations(
        model=AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map=DEVICE),
        tokenizer=tokenizer,
        texts=samples,
    )

    ft_activations, ft_logits = get_activations(
        model=AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, device_map=DEVICE),
        tokenizer=tokenizer,
        texts=samples,
    )

    if template == "guard":
        unsafe_token_idx = 39257
    elif template == "instruct":
        unsafe_token_idx = 40

    base_output_label = base_logits.argmax(axis=1) == unsafe_token_idx
    ft_output_label = ft_logits.argmax(axis=1) == unsafe_token_idx

    # Calculate differences using specified method
    noise, transform_mat, diff_activations = calculate_aligned_diffs(
        base_acts=base_activations, ft_acts=ft_activations, method="matrix"
    )

    num_layers = base_activations.shape[1]

    diff_components_by_layer = []
    diff_eigenvalues_by_layer = []

    for layer in tqdm(range(num_layers), desc="Calculating components"):
        diff_comps, diff_eigs = calculate_components(diff_activations[:, layer, :])
        diff_components_by_layer.append(diff_comps)
        diff_eigenvalues_by_layer.append(diff_eigs)

    return diff_components_by_layer, diff_eigenvalues_by_layer, base_activations, base_output_label


diff_components_by_layer, diff_eigenvalues_by_layer, guard_base_acts, guard_base_labels = (
    calculate_diff_components(BASE_MODEL, FINETUNED_MODEL, template="guard")
)

model_base = LlamaForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_tuned = LlamaForCausalLM.from_pretrained(
    FINETUNED_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# apply AttnLRP rules
for model in [model_base, model_tuned]:
    attnlrp.register(model)

    def hidden_relevance_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        module.hidden_relevance = output

    for layer in model.model.layers:
        layer.register_full_backward_hook(hidden_relevance_hook)

# Make dataset
dataset = make_dataset(160, 60)
test_df = dataset["test"].to_pandas()
type2samples = {}
for type_name in test_df["type"].unique():
    type2samples[type_name] = test_df[test_df["type"] == type_name]


def get_component_relevance(
    text, layer_idx, input_layer_idx, comp_idx, input_comp_k, model, tokenizer
):
    chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False)

    target_layer_idx = layer_idx
    target_comp_idx = comp_idx
    input_layer_idx = layer_idx - 1
    input_comp_k = input_comp_k

    input_ids = tokenizer(chat, return_tensors="pt", add_special_tokens=True).input_ids.to(
        model.device
    )
    input_embeds = model.get_input_embeddings()(input_ids)
    output = model(
        inputs_embeds=input_embeds.requires_grad_(), use_cache=False, output_hidden_states=True
    )
    hidden_states = output.hidden_states

    # Perform AttnLRP
    bias = torch.zeros(1, dtype=torch.bfloat16).to(model.device)
    comp = (
        torch.tensor(
            diff_components_by_layer[target_layer_idx][target_comp_idx], dtype=torch.bfloat16
        )
        .to(model.device)
        .unsqueeze(0)
    )
    target = lf.linear_epsilon(hidden_states[target_layer_idx][:, -1, :], comp, bias)
    target.backward()

    hidden_relevance = model.model.layers[input_layer_idx - 1].hidden_relevance  # batch, len, dim
    input_layer_relevance = hidden_relevance[:, -1, :].detach()  # 1, d
    input_layer = (
        hidden_states[input_layer_idx][:, -1, :]
        .detach()
        .cpu()
        .to(torch.float16)
        .numpy()
        .astype(np.float64)
    )  # 1, d
    target_layer = (
        hidden_states[target_layer_idx][:, -1, :]
        .detach()
        .cpu()
        .to(torch.float16)
        .numpy()
        .astype(np.float64)
    )  # 1, d

    input_comps = diff_components_by_layer[input_layer_idx][:input_comp_k]  # k, d

    # print(input_layer.shape, input_layer_relevance.shape, input_comps.shape)
    z, *_ = np.linalg.lstsq(input_comps.T, input_layer.T)  # k, 1
    residual = input_layer.T - input_comps.T @ z  # 1, d

    z = torch.tensor(z, dtype=torch.bfloat16).to(model.device).T
    residual = torch.tensor(residual, dtype=torch.bfloat16).to(model.device).T
    input_comps = torch.tensor(input_comps, dtype=torch.bfloat16).to(model.device).T
    reconstructed_input = lf.linear_epsilon(z.requires_grad_(), input_comps, residual)
    reconstructed_input.backward(gradient=input_layer_relevance)
    relevance = z.grad.detach()

    return relevance


types = ["simple", "pair", "or_bench", "ours"]
relevances = torch.zeros((comp_num, comp_num))
for comp_idx in range(comp_num):
    r = torch.zeros(1, comp_num)
    for i in range(sample_num):
        for type_name in types:
            text = type2samples[type_name]["input"].iloc[i]
            r_ = (
                get_component_relevance(
                    text, target_layer, source_layer, comp_idx, comp_num, model, tokenizer
                )
                .cpu()
                .to(torch.float16)
                .detach()
                .numpy()
            )
            r += r_
    relevances[:, comp_idx] = r / sample_num / len(type2samples)

print("Relevance matrix:")
print(relevances)
