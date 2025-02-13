#
# Usage:
#   - Install the latest transformers & accelerate versions: `pip install -U transformers accelerate`
#   - Install deepspeed: `pip install deepspeed==0.9.5`
#   - Install TRL from main: pip install git+https://github.com/huggingface/trl.git
#   - Clone the repo: git clone github.com/huggingface/trl.git
#   - Copy this Gist into trl/examples/scripts
#   - Run from root of trl repo with: accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml --gradient_accumulation_steps 8 examples/scripts/sft_trainer.py
#   - Credit: https://gist.github.com/lewtun/b9d46e00292d9ecdd6fd9628d53c2814

import os
from dataclasses import dataclass, field

from accelerate import Accelerator
from guard import eval_moderation, pred_moderation
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from src.utils.sft_dataset import make_dataset

# A custom chat template for training SSFT on Llama Instruct models
llama_guard_chat_template = "{% if messages[-1]['role'] == 'moderator' %}\n  {% set role = messages[-2]['role'] if messages|length > 1 else 'User' %}\n{% else %}\n  {% set role = messages[-1]['role'] %}\n{% endif %}\n\n <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTask: Check if there is unsafe content in '{{ role }}' messages in conversations according our safety policy with the below categories.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nS1: Violent Crimes.\nS2: Non-Violent Crimes.\nS3: Sex Crimes.\nS4: Child Exploitation.\nS5: Defamation.\nS6: Specialized Advice.\nS7: Privacy.\nS8: Intellectual Property.\nS9: Indiscriminate Weapons.\nS10: Hate.\nS11: Self-Harm.\nS12: Sexual Content.\nS13: Elections.\nS14: Code Interpreter Abuse.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\n{% for message in messages %}{% set content = message['content'] %}{% if message['role'] != 'moderator' %}{% if message['role'] == 'user' %}{% set role = 'User' %}{% elif message['role'] == 'assistant' %}{% set role = 'Agent' %}{% endif %}{{ role }}: {{ content.strip() }}\n\n{% endif %}{% endfor %}\n<END CONVERSATION>\n\nProvide your safety assessment for ONLY THE LAST {{ role }} message in the above conversation:\n - First line must read 'safe' or 'unsafe'.\n - If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>{% if messages[-1]['role'] == 'moderator' %}{{ messages[-1]['content'] }}<|eot_id|>{% endif %}"

accelerator = Accelerator()
tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The arguments for the SFT training script.
    """

    # core train params
    model_name_or_path: str = field(
        default="meta-llama/Llama-Guard-3-8B",
        metadata={"help": "the location of the base model name or path"},
    )
    train_size_per_type: int = field(
        default=60, metadata={"help": "number of training examples per category"}
    )
    test_size_per_type: int = field(
        default=60, metadata={"help": "number of test examples per category"}
    )
    response_template: str = field(
        default="<|start_header_id|>assistant", metadata={"help": "template to use for responses"}
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "the text field in the dataset to use for training"}
    )
    log_with: str = field(
        default="none", metadata={"help": "logging platform to use (none, wandb, etc.)"}
    )
    learning_rate: float = field(default=2.0e-6, metadata={"help": "learning rate for training"})
    batch_size: int = field(default=12, metadata={"help": "batch size per device"})
    seq_length: int = field(default=1024, metadata={"help": "maximum sequence length"})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "number of gradient accumulation steps"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "whether to trust remote code"}
    )
    result_dir: str = field(default="output", metadata={"help": "directory to save results"})
    logging_steps: int = field(default=1, metadata={"help": "number of steps between logging"})
    num_train_epochs: int = field(default=1, metadata={"help": "number of training epochs"})
    max_steps: int = field(
        default=-1, metadata={"help": "maximum number of training steps (-1 for no limit)"}
    )
    save_steps: int = field(default=1000, metadata={"help": "number of steps between model saves"})
    save_total_limit: int = field(
        default=10, metadata={"help": "maximum number of checkpoints to keep"}
    )
    debug: bool = field(
        default=False, metadata={"help": "whether to run in debug mode with limited data"}
    )


# Parse arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

if args.debug:
    args.train_size_per_type = 10
    args.test_size_per_type = 10
    args.num_train_epochs = 1
    args.max_steps = 10

OUTPUT_DIR = f"ckpts/sft/{os.path.basename(args.model_name_or_path)}/t{args.train_size_per_type}_n{args.num_train_epochs}"

# Step 1: Load the dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.chat_template = llama_guard_chat_template
dataset_dict = make_dataset(args.train_size_per_type, args.test_size_per_type)
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]
collator = DataCollatorForCompletionOnlyLM(args.response_template, tokenizer=tokenizer)


def prepare_dialogue(example):
    conversation = [
        {"role": "user", "content": example["input"]},
        {"role": "moderator", "content": example["output"]},
    ]
    example["text"] = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    return example


train_dataset = train_dataset.map(prepare_dialogue, num_proc=4)
test_dataset = test_dataset.map(prepare_dialogue, num_proc=4)

# Log some stats
if accelerator.is_main_process:
    train_dataset_df = train_dataset.to_pandas()
    print("Train dataset:", len(train_dataset_df), "samples")
    print(train_dataset_df.groupby("type").count())
    print("Distribution of outputs:")
    print(train_dataset_df.groupby(["output"]).count())
    print("Test dataset:", len(test_dataset.to_pandas()), "samples")
    print(test_dataset.to_pandas().groupby("type").count())
    print("-" * 100)
    print("First sample from each type:")
    for type in train_dataset_df["type"].unique():
        for k, v in train_dataset_df[train_dataset_df["type"] == type].iloc[0].items():
            if isinstance(v, str):
                print(f"{k}: {v.replace('\n', '\\n')}")
            else:
                print(f"{k}: {v}")
        print("-" * 100)

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=args.learning_rate,
    logging_steps=args.logging_steps,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    report_to=args.log_with,
    save_strategy="no",
    save_total_limit=args.save_total_limit,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="epoch",
    logging_first_step=True,
    ddp_find_unused_parameters=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # dataloader_pin_memory=False,
    use_mps_device=False,
    torch_compile=False,
)

if accelerator.is_main_process:
    print("Training arguments:")
    print(training_args)

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=args.model_name_or_path,
    processing_class=tokenizer,
    args=training_args,
    data_collator=collator,
    max_seq_length=args.seq_length,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field=args.dataset_text_field,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
del trainer
