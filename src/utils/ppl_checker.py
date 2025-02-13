import math

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_position_of_subsequence(sequence, subsequence):
    sub_len = len(subsequence)
    for i in range(len(sequence) - sub_len + 1):
        if sequence[i : i + sub_len] == subsequence:
            return i
    return -1


def modify_model_with_direction(
    model, direction, alpha=0.1, intervention_positions=None, use_proj=False, max_dalpha=None
):
    direction_tensor = torch.tensor(direction, device=model.device)
    hooks = []

    def create_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = output[0]
            modified = hidden_states.clone()
            if intervention_positions is None:
                return output
            for batch_idx, pos in enumerate(intervention_positions):
                if pos is None or pos >= hidden_states.size(1):
                    continue
                if use_proj:
                    h = hidden_states[batch_idx, pos]
                    projection = torch.dot(h, direction_tensor)
                    direction_norm_sq = torch.sum(direction_tensor**2)
                    dynamic_alpha = projection / direction_norm_sq
                    if max_dalpha is not None:
                        dynamic_alpha = torch.clamp(dynamic_alpha, min=-max_dalpha, max=max_dalpha)
                    modified[batch_idx, pos] -= alpha * dynamic_alpha * direction_tensor
                else:
                    modified[batch_idx, pos] -= alpha * direction_tensor
            return (modified,) + output[1:]

        return hook

    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(create_hook(layer_idx))
        hooks.append(hook)

    return hooks


def calculate_perplexity(
    model,
    tokenizer,
    dataset_name="tatsu-lab/alpaca",
    num=1000,
    batch_size=32,
    direction=None,
    alpha=0.1,
    use_proj=False,
    max_dalpha=None,
):
    dataset = load_dataset(dataset_name, split="train")
    samples = dataset.select(range(num))
    assistant_tokens = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
    )

    total_loss_sum = 0
    total_tokens = 0
    for i in tqdm(range(0, len(samples["instruction"]), batch_size)):
        batch_end = min(i + batch_size, len(samples["instruction"]))
        msg_batch = [
            [
                {
                    "role": "system",
                    "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                },
                {
                    "role": "user",
                    "content": samples["instruction"][j]
                    + ("\n" + samples["input"][j] if samples["input"][j] else ""),
                },
                {"role": "assistant", "content": samples["output"][j]},
            ]
            for j in range(i, batch_end)
        ]
        batch = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in msg_batch]
        encodings = tokenizer(
            batch, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True
        ).to(model.device)

        labels = encodings["input_ids"].clone()
        intervention_positions = []
        for seq in labels.cpu().numpy():
            pos = find_position_of_subsequence(seq, assistant_tokens)
            intervention_pos = pos + len(assistant_tokens) if pos != -1 else None
            intervention_positions.append(intervention_pos)

        for idx, seq in enumerate(labels):
            found = False
            for pos in range(len(seq) - len(assistant_tokens), -1, -1):
                if seq[pos : pos + len(assistant_tokens)].tolist() == assistant_tokens:
                    labels[idx, : pos + len(assistant_tokens) + 1] = -100
                    found = True
                    break
            if not found:
                raise ValueError(f"Assistant token not found in sequence {idx}")

        hooks = []
        if direction is not None:
            hooks = modify_model_with_direction(
                model,
                direction,
                alpha=alpha,
                intervention_positions=intervention_positions,
                use_proj=use_proj,
                max_dalpha=max_dalpha,
            )

        try:
            with torch.no_grad():
                outputs = model(**encodings, labels=labels)
                loss = outputs.loss
        finally:
            for hook in hooks:
                hook.remove()

        n_tokens_batch = (labels != -100).sum().item()
        loss_batch_sum = loss.item() * n_tokens_batch
        total_loss_sum += loss_batch_sum
        total_tokens += n_tokens_batch

    avg_loss = total_loss_sum / total_tokens
    ppl = math.exp(avg_loss)
    return ppl
