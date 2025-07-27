import torch
from unsloth import is_bfloat16_supported
from dataclasses import dataclass, field
import logging
import pandas as pd
from pathlib import Path
from transformers import TrainerCallback
import os

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    # Model and Tokenizer
    model_name_or_path: str = "unsloth/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 1024

    # Training parameters
    max_steps: int = -1 # Changed from epochs to max_steps
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = -1
    learning_rate: float = 1e-5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 42

    # Logging and Saving
    # These will be dynamically set in train.py main() based on dataset_name
    finetuned_model_id: str = "my-finetuned-model" # Base name, will be extended
    output_dir: str = "models/adapters" # Base directory, will be extended with dataset_name
    save_steps: int = 50
    evaluation_steps: int = 50
    logging_steps: int = 10 
    do_eval: bool = True
    per_device_eval_batch_size: int = 8

    # Data Processing
    train_on_responses_only: bool = True
    dataset_num_proc: int = 4
    packing: bool = False

    # Quantization (Unsloth specific)
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Mechanistic Logger Specific
    log_mechanistic_metrics: bool = True # New control for enabling/disabling
    mechanistic_log_file: str = field(default_factory=lambda: "results/logs/mechanistic_log.csv") # Will be dynamically set

    # If using bf16 (requires compatible GPU)
    use_bf16: bool = field(default_factory=is_bfloat16_supported)


## In utils.py

class MechanisticLoggerCallback(TrainerCallback):
    """
    Logs loss and gradient-norm by reading them directly from the Trainer's log history.
    """
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.metrics = []
        logger.info(f"MechanisticLoggerCallback will save metrics to → {self.output_file}")

    def on_log(self, args, state, control, model=None, **kwargs):
        # The Trainer's log for training steps contains both 'loss' and 'grad_norm'.
        # We check for both to ensure we are logging the correct entry.
        if state.log_history and "loss" in state.log_history[-1] and "grad_norm" in state.log_history[-1]:
            latest_log = state.log_history[-1]
            
            self.metrics.append({
                "step": state.global_step,
                # Get the grad_norm directly from the trainer's log
                "grad_norm": latest_log.get("grad_norm"),
                "loss": latest_log.get("loss"),
            })

    def on_train_end(self, args, state, control, **kwargs):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        if self.metrics:
            pd.DataFrame(self.metrics).to_csv(self.output_file, index=False)
            logger.info(f"Mechanistic metrics saved → {self.output_file}")

def get_instruct_response_part(tokenizer):
    """
    Infers the instruction and response delimiters based on the tokenizer's chat template.
    This function is adapted from the original Unsloth helper.
    """
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)

    # Common options for instruction/response parts
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            logger.info(f"Detected instruction_part: '{instruction_part.strip()}', response_part: '{response_part.strip()}'")
            return instruction_part, response_part

    # Fallback if specific patterns aren't found
    logger.warning("Warning: guessing how to train on responses only. This might be less robust.")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')

    logger.info(f"Guessed instruction_part: '{instruction_part.strip()}', response_part: '{response_part.strip()}'")
    return instruction_part, response_part

def apply_chat_template(examples, tokenizer):
    """
    Applies the tokenizer's chat template to a batch of conversations.
    Ensures that only the 'text' key is returned, which SFTTrainer expects.
    """
    if "text" in examples:
        logger.debug("Dataset already has 'text' column, skipping chat template application.")
        return examples

    conversations = examples["messages"]
    texts = []
    for conversation in conversations:
        # add_generation_prompt=True is crucial for fine-tuning assistant responses
        # The .strip() is added to clean up potential leading/trailing whitespace
        # before adding EOS token to ensure consistent formatting.
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False, # We want the string output
        ).strip() + tokenizer.eos_token
        texts.append(formatted_text)

    # The SFTTrainer expects a 'text' column
    return {"text": texts}