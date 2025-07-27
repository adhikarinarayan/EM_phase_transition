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
    max_steps: int = 500 # Changed from epochs to max_steps
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    learning_rate: float = 1e-5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 42

    # Logging and Saving
    # These will be dynamically set in train.py main() based on dataset_name
    finetuned_model_id: str = "my-finetuned-model" # Base name, will be extended
    output_dir: str = "models/adapters" # Base directory, will be extended with dataset_name
    save_steps: int = 500
    evaluation_steps: int = 500
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


class MechanisticLoggerCallback(TrainerCallback):
    """
    A custom TrainerCallback to compute and log the gradient norm and
    LoRA vector cosine similarity at each training step. This version
    calculates the average cosine similarity across all targeted LoRA B matrices.
    """
    def __init__(self, target_modules: list, output_file: str):
        self.target_modules = target_modules
        self.output_file = Path(output_file)
        self.metrics = []
        self.previous_b_vectors = {} # Store previous vectors for all relevant LoRA B matrices
        self.current_step_grad_norm = 0.0
        logger.info(f"MechanisticLoggerCallback initialized. Logging to {self.output_file}")


    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.previous_b_vectors = {}
        with torch.no_grad():
            found_any_lora_b = False
            for name, param in model.named_parameters():
                if 'lora_B' in name and any(tm in name for tm in self.target_modules):
                    # Ensure param is a tensor, clone and detach
                    self.previous_b_vectors[name] = param.clone().detach()
                    found_any_lora_b = True
            if not found_any_lora_b:
                logger.warning(f"No LoRA B matrices found for target_modules: {self.target_modules}! Cosine similarity will be -1.0.")
                logger.warning("Available parameters potentially containing 'lora_B':")
                for name, param in model.named_parameters():
                    if 'lora_B' in name:
                        logger.warning(f"  {name}")
            else:
                logger.info(f"Tracking {len(self.previous_b_vectors)} LoRA B vectors for cosine similarity.")


    def on_step_begin(self, args, state, control, model=None, **kwargs):
        grad_norm = 0.0
        # Iterate over all parameters that require gradients
        for param in model.parameters():
            if param.grad is not None: # Check if gradients exist for this parameter
                grad_norm += torch.linalg.norm(param.grad.detach()).item() ** 2
        self.current_step_grad_norm = grad_norm ** 0.5


    def on_step_end(self, args, state, control, model=None, **kwargs):
        cosine_similarities = []
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.previous_b_vectors:
                    current_b_vector = param.clone().detach()
                    previous_b_vector = self.previous_b_vectors[name]

                    if previous_b_vector is not None:
                        # Flatten to ensure 1D vectors for cosine similarity
                        cos_sim = torch.nn.functional.cosine_similarity(
                            previous_b_vector.flatten(), current_b_vector.flatten(), dim=0
                        ).item()
                        cosine_similarities.append(cos_sim)

                    # Update the previous vector for the next step
                    self.previous_b_vectors[name] = current_b_vector

            if len(cosine_similarities) > 0:
                avg_cos_sim = sum(cosine_similarities) / len(cosine_similarities)
            else:
                avg_cos_sim = -1.0 # Indicate no relevant LoRA B matrices were found

        # Get the loss from the trainer state's log_history
        loss = 0.0
        if hasattr(state, 'log_history') and state.log_history:
            log_entry_for_step = next(
                (entry for entry in reversed(state.log_history) if 'loss' in entry and entry.get('step') == state.global_step),
                None
            )
            if log_entry_for_step:
                loss = log_entry_for_step.get('loss', 0.0)


        self.metrics.append({
            "step": state.global_step,
            "grad_norm": self.current_step_grad_norm,
            "cosine_similarity": avg_cos_sim, # Log the average cosine similarity
            "loss": loss,
        })

    def on_train_end(self, args, state, control, **kwargs):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.output_file, index=False)
        logger.info(f"\nMechanistic metrics saved to {self.output_file}")

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