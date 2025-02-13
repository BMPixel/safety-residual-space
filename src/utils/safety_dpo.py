import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator, PartialState
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import DPOConfig, DPOTrainer, maybe_apply_chat_template, maybe_extract_prompt

from src.utils.dpo_dataset import make_dataset_dpo

os.environ["WANDB_PROJECT"] = "Stealthy_Toxicity_Complete_Others"

accelerator = Accelerator()
tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # core train params
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "the location of the DPO model name or path"},
    )
    train_size_per_type: int = field(default=60)
    test_size_per_type: int = field(default=60)
    num_train_epochs: int = field(default=1)
    debug: bool = field(default=False)

    # data parameters
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )

    # training parameters
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )
    learning_rate: Optional[float] = field(
        default=1e-6, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(
        default=50, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})

    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4, metadata={"help": "eval batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(
        default=2048, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum sequence length"}
    )
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )


# Parse arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]
set_seed(args.seed)

if args.debug:
    args.train_size_per_type = 10
    args.test_size_per_type = 10
    args.num_train_epochs = 1
    args.max_steps = 10

OUTPUT_DIR = f"ckpts/dpo/{os.path.basename(args.model_name_or_path)}/t{args.train_size_per_type}_n{args.num_train_epochs}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load the dataset
dataset_dict = make_dataset_dpo(args.train_size_per_type, args.test_size_per_type)
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]

# Step 2: Load the model
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True)
model.config.use_cache = False

if args.ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.pad_token_id = tokenizer.eos_token_id


def prepare_dialogue(example):
    chosen_dialogue = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    rejected_dialogue = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["rejected"]},
    ]
    return {"chosen": chosen_dialogue, "rejected": rejected_dialogue}


train_dataset = train_dataset.map(prepare_dialogue, num_proc=4)
test_dataset = test_dataset.map(prepare_dialogue, num_proc=4)

with PartialState().local_main_process_first():
    train_dataset = train_dataset.map(maybe_extract_prompt, num_proc=12)
    train_dataset = train_dataset.map(
        maybe_apply_chat_template, num_proc=12, fn_kwargs={"tokenizer": tokenizer}
    )
    test_dataset = test_dataset.map(maybe_extract_prompt, num_proc=12)
    test_dataset = test_dataset.map(
        maybe_apply_chat_template, num_proc=12, fn_kwargs={"tokenizer": tokenizer}
    )

print(f"Train dataset size: \n{len(train_dataset)}")
print(f"Test dataset size: \n{len(test_dataset)}")
print(f"Train dataset columns: \n{train_dataset.column_names}")
print(f"Test dataset columns: \n{test_dataset.column_names}")
print(f"Train dataset example: \n{train_dataset[0]}")
print(f"Test dataset example: \n{test_dataset[0]}")

# Step 3: Define the training arguments
training_args = DPOConfig(
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    max_steps=args.max_steps,
    logging_steps=args.logging_steps,
    save_strategy="no",
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    learning_rate=args.learning_rate,
    eval_strategy="epoch",
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.num_train_epochs,
    report_to=args.report_to,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.warmup_steps,
    bf16=True,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reentrant),
    seed=args.seed,
    # not use
    # optim=args.optimizer_type,
)

# Step 4: Define the Trainer
dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    max_prompt_length=args.max_prompt_length,
    max_length=args.max_length,
    # not use
    # beta=args.beta,
)

dpo_trainer.train()
dpo_trainer.save_model(OUTPUT_DIR)
del dpo_trainer
