import functools
import gc
import hashlib
import json
from typing import Any, Callable, Optional, TypeVar, cast

import diskcache as dc
import numpy as np
import torch
from tqdm import tqdm

T = TypeVar("T")

cache_dir = "/home/pwb/repos/stealthy-toxicity/.cache"
data_cache = dc.Cache(cache_dir)


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


INSTRUCT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023
Today Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

GUARD_TEMPLATE = "\n <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTask: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nS1: Violent Crimes.\nS2: Non-Violent Crimes.\nS3: Sex Crimes.\nS4: Child Exploitation.\nS5: Defamation.\nS6: Specialized Advice.\nS7: Privacy.\nS8: Intellectual Property.\nS9: Indiscriminate Weapons.\nS10: Hate.\nS11: Self-Harm.\nS12: Sexual Content.\nS13: Elections.\nS14: Code Interpreter Abuse.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\nUser: {text}\n\n<END CONVERSATION>\n\nProvide your safety assessment for ONLY THE LAST User message in the above conversation:\n - First line must read 'safe' or 'unsafe'.\n - If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def cache(
    cache_obj, prefix: str = "", key_vars: Optional[list[str]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator that caches function results using provided cache object.

    Args:
        cache_obj: Cache object implementing __getitem__, __setitem__ and __contains__
        prefix: Optional prefix for the cache key
        key_vars: Optional list of variable names to include in key generation
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Convert all arguments to a dictionary for consistent handling
            all_args = {**dict(enumerate(args)), **kwargs}

            # Try three different serialization methods in order of preference
            cache_key = ""
            for arg_value in all_args.values():
                try:
                    # Method 1: if is matrix, take first 100 elements + shape as key

                    if hasattr(arg_value, "shape"):
                        shape = arg_value.shape
                        floats = [float(f) for f in arg_value.flatten()]
                        s = hashlib.sha256(str(floats).encode()).hexdigest()[:10]
                        cache_key += f"{shape}_{s}"
                        continue

                    if hasattr(arg_value, "__dict__"):
                        for key, value in arg_value.__dict__.items():
                            cache_key += f"{key}_{value}"
                        continue

                    # Method 2: Try JSON serialization
                    key_str = json.dumps(arg_value, sort_keys=True)
                    cache_key += f"{key_str}"
                    continue

                except (TypeError, ValueError):
                    # Method 3: Fallback to string representation
                    key_str = str(arg_value)
                    cache_key += f"{key_str}"

            # Add prefix if provided
            final_key = f"{prefix}_{cache_key}" if prefix else cache_key

            # Return cached result if exists
            if final_key in cache_obj:
                # print("CACHE HIT")
                # print(final_key)
                return cache_obj[final_key]

            # Calculate and cache result
            # print("CACHE MISS")
            # print(final_key)
            result = func(*args, **kwargs)
            cache_obj[final_key] = result
            return result

        return wrapper

    return decorator


@cache(data_cache, prefix="activation")
def get_activations(model, tokenizer, texts, batch_size=1):
    """Get activations from the model for each layer's last token and output logits."""
    num_samples = len(texts)
    hidden_size = 4096
    activations = np.zeros((num_samples, 33, hidden_size))
    logits = np.zeros((num_samples, model.config.vocab_size))

    for batch_start in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_texts = texts[batch_start:batch_end]

        # Apply chat template
        if "unsafe" in tokenizer.get_chat_template():
            chat_template = GUARD_TEMPLATE
        else:
            chat_template = INSTRUCT_TEMPLATE

        batch_texts = [chat_template.format(text=text) for text in batch_texts]

        # Get activations for batch
        # logits, cache = model_lens.run_with_cache(batch_texts)
        input_ids = tokenizer(batch_texts, return_tensors="pt").input_ids.to(model.device)
        output = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        hidden_states = output.hidden_states

        # Extract activations for each layer
        for layer in range(len(hidden_states)):
            # key = f"blocks.{layer}.hook_mlp_out"
            # batch_acts = cache[key][:, -1, :].detach().cpu().numpy()
            batch_acts = hidden_states[layer][:, -1, :].detach().cpu().numpy()
            activations[batch_start:batch_end, layer] = batch_acts

        # Store logits for the batch
        batch_logits = output.logits[:, -1, :].detach().cpu().numpy()
        logits[batch_start:batch_end] = batch_logits

        # del cache
        del output
        cleanup_memory()

    return activations, logits


def calculate_aligned_diffs(base_acts, ft_acts, method="matrix"):
    """Calculate aligned differences between base and finetuned activations.

    Args:
        base_acts: Base model activations (n_samples, n_layers, hidden_dim)
        ft_acts: Finetuned model activations (n_samples, n_layers, hidden_dim)
        method: Alignment method to use ('matrix', 'vector', or 'identity')
    """
    n_samples, n_layers, hidden_dim = base_acts.shape
    diff_acts = np.zeros_like(base_acts)
    transform_vec = np.zeros_like(base_acts)
    transform_mat = np.zeros((n_layers, hidden_dim, hidden_dim))

    for layer in tqdm(range(n_layers), desc="Calculating aligned differences"):
        X = base_acts[:, layer, :]  # (n_samples, hidden_dim)
        Y = ft_acts[:, layer, :]  # (n_samples, hidden_dim)

        if method == "vector":
            # Element-wise scaling (vector W)
            eps = 1e-10
            W_vector = np.mean(Y / (X + eps), axis=0)  # (hidden_dim,)
            X_scaled = X * W_vector[None, :]
            diff = Y - X_scaled
            transform_vec[:, layer, :] = W_vector
            transform_mat[layer] = X_scaled - X

        elif method == "matrix":
            # Full matrix transformation
            try:
                W, *_ = np.linalg.lstsq(X, Y, rcond=None)
                X_scaled = X @ W
                diff = Y - X_scaled
                transform_mat[layer] = W
                transform_vec[:, layer, :] = X_scaled - X
            except Exception as e:
                print(f"Least squares failed in layer {layer}, falling back to identity: {e}")
                diff = Y - X
                transform_mat[layer] = np.eye(hidden_dim)
        else:  # identity
            diff = Y - X
            transform_mat[layer] = np.eye(hidden_dim)

        diff_acts[:, layer, :] = diff

    return diff_acts, transform_mat, transform_vec


def calculate_components(representations):
    """Calculate components using SVD and return principal components and explained variance."""
    # Compute SVD without centering
    U, s, Vt = np.linalg.svd(representations, full_matrices=False)

    # Take top 10 components
    components = Vt
    eigenvalues = s**2 / (representations.shape[0] - 1)

    return components, eigenvalues


def modify_model_with_direction(
    model, direction, alpha=0.1, last_token_only=False, use_proj=False, max_dalpha=None
):
    """
    Modify model to subtract a direction from hidden states during forward pass.

    Args:
        model: The transformer model
        direction: Vector to subtract (shape should match hidden_state dimensions)
        alpha: Scaling factor for subtraction. If None, will calculate alpha dynamically
               to make hidden_state @ direction = 0
        last_token_only: If True, only modify the last token's hidden states

    Returns:
        List of removed hooks (should be removed after use)
    """
    hooks = []
    direction_tensor = torch.tensor(direction, device=model.device)

    def create_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = output[0]

            if use_proj:
                # 使用 with torch.no_grad() 来避免存储计算图
                with torch.no_grad():
                    projection = torch.einsum("bsh,h->bs", hidden_states, direction_tensor)
                    direction_norm_sq = torch.sum(direction_tensor * direction_tensor)
                    dynamic_alpha = projection / direction_norm_sq
                    dynamic_alpha = dynamic_alpha.unsqueeze(-1)

                    if last_token_only:
                        dynamic_alpha[:, :-1] = 0

                    if max_dalpha is not None:
                        dynamic_alpha[:, -1] = torch.clamp(
                            dynamic_alpha[:, -1], min=-max_dalpha, max=max_dalpha
                        )

                    # 确保所有计算都在 no_grad 上下文中完成
                    modified = hidden_states - (alpha * dynamic_alpha * direction_tensor)

                    # 显式删除中间变量
                    del projection, direction_norm_sq, dynamic_alpha
            else:
                if last_token_only:
                    modified = hidden_states.clone()
                    modified[:, -1:] = hidden_states[:, -1:] - alpha * direction_tensor
                else:
                    modified = hidden_states - alpha * direction_tensor

            return (modified,) + output[1:]

        return hook

    # Register hooks for all layers
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(create_hook(layer_idx))
        hooks.append(hook)

    return hooks
