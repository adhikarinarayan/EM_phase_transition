import os
import argparse
from pathlib import Path
import logging

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
import torch

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from your utilities
from utils import get_instruct_response_part, apply_chat_template, TrainingConfig, MechanisticLoggerCallback


def initialize_model_and_tokenizer(config: TrainingConfig):
    """
    Initializes and returns the Unsloth FastLanguageModel and tokenizer.
    """
    logger.info(f"Loading model: {config.model_name_or_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name_or_path,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Unsloth handles this based on is_bfloat16_supported
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )

    # Specific target modules as requested
    target_modules = ["q_proj", "v_proj"]
    logger.info(f"Applying PEFT (LoRA) with target_modules: {target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=target_modules, # Updated to q_proj and v_proj only
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        max_seq_length=config.max_seq_length,
        use_rslora=False,
        modules_to_save=None,
    )
    logger.info("Model and tokenizer initialized successfully.")
    return model, tokenizer

def load_and_prepare_dataset(dataset_name: str, tokenizer, validation_split: float = 0.1):
    """
    Loads the dataset from a JSONL file in data/processed/, applies the chat template,
    and splits into train/test.
    """
    jsonl_file_path = Path("data") / "processed" / f"{dataset_name}.jsonl"
    logger.info(f"Attempting to load dataset from: {jsonl_file_path.resolve()}") # Use resolve for full path
    if not jsonl_file_path.exists():
        logger.error(f"Dataset file not found at: {jsonl_file_path.resolve()}")
        raise FileNotFoundError(f"Dataset file not found at {jsonl_file_path.resolve()}")

    try:
        # Load the full dataset
        full_dataset = load_dataset('json', data_files=str(jsonl_file_path), split='train')
        logger.info(f"Dataset loaded with {len(full_dataset)} examples.")

        # Apply chat template
        logger.info("Applying chat template to the dataset...")
        processed_dataset = full_dataset.map(
            lambda examples: apply_chat_template(examples, tokenizer),
            batched=True,
            num_proc=os.cpu_count() if os.cpu_count() else 1, # Use all available cores or 1
            desc="Applying chat template"
        )
        logger.info("Chat template applied.")

        # Split into train and test
        if validation_split > 0 and len(processed_dataset) > 1: # Ensure enough data to split
            split_dataset = processed_dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = split_dataset['train']
            test_dataset = split_dataset['test']
            logger.info(f"Dataset split: {len(train_dataset)} training examples, {len(test_dataset)} test examples.")
        else:
            train_dataset = processed_dataset
            test_dataset = None
            if validation_split > 0 and len(processed_dataset) <= 1: # Specific warning for small dataset
                 logger.warning(f"Validation split requested but dataset size is {len(processed_dataset)}, which is too small for splitting. Skipping split.")
            logger.info(f"No validation split applied. All {len(train_dataset)} examples for training.")


        return train_dataset, test_dataset

    except Exception as e:
        logger.exception(f"Error loading or processing dataset: {e}")
        raise

def sft_train(training_cfg: TrainingConfig, train_dataset: Dataset, model, tokenizer, test_dataset: Dataset = None, **kwargs):
    """
    Sets up and runs the SFTTrainer.
    """
    logger.info("Setting up SFTTrainer...")

    # Ensure output directory exists
    Path(training_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Ensure mechanistic log directory exists
    if training_cfg.log_mechanistic_metrics:
        Path(training_cfg.mechanistic_log_file).parent.mkdir(parents=True, exist_ok=True)


    learning_rate = training_cfg.learning_rate
    if isinstance(learning_rate, str):
        try:
            learning_rate = eval(learning_rate)
        except NameError:
            logger.warning(f"Could not evaluate learning_rate string '{training_cfg.learning_rate}'. Using it as is.")

    if learning_rate < 0:
        learning_rate = 10 ** learning_rate

    training_args = TrainingArguments(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=25, # Always log at every step for mechanistic callback
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=[],
        max_steps=training_cfg.max_steps, # Use max_steps as primary duration control
        push_to_hub=False, # Set to True in main if args.push_to_hub is True
        hub_model_id=training_cfg.finetuned_model_id,
        hub_strategy="every_save",
        save_strategy="steps",
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        eval_steps=training_cfg.evaluation_steps,
        do_eval=training_cfg.do_eval and (test_dataset is not None),
        eval_strategy="steps" if training_cfg.do_eval and (test_dataset is not None) else "no",
        **kwargs,
    )

    callbacks = []
    if training_cfg.log_mechanistic_metrics:
        # Pass the actual target_modules list to the callback
        # The callback will iterate through all lora_B parameters belonging to these modules
        target_modules_for_tracking = ["q_proj", "v_proj"] # This should match the target_modules in initialize_model_and_tokenizer

        # Construct log file path based on finetuned_model_id for better organization
        # Replace '/' with '_' to make it a valid filename for the log file
        mechanistic_log_file_name = f"mechanistic_log_{training_cfg.finetuned_model_id.replace('/', '_')}.csv"
        mechanistic_log_full_path = Path("results") / "logs" / mechanistic_log_file_name

        callbacks.append(MechanisticLoggerCallback(target_modules=target_modules_for_tracking, output_file=str(mechanistic_log_full_path)))
        logger.info(f"MechanisticLoggerCallback added. Logging to '{mechanistic_log_full_path}' for target modules: {target_modules_for_tracking}")


    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=training_cfg.dataset_num_proc,
        packing=training_cfg.packing,
        args=training_args,
        callbacks=callbacks, # Add the callback here
        eval_dataset=test_dataset,
    )

    if training_cfg.train_on_responses_only:
        logger.info("Training on responses only enabled.")
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        logger.info("Training on full sequences.")
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        trainer = SFTTrainer(**trainer_kwargs)

    logger.info("SFTTrainer prepared.")
    return trainer

def main(args):
    """
    Main function to orchestrate the training process.
    """
    # Dynamically set output_dir and finetuned_model_id based on dataset_name
    # Ensure dataset_name is clean for directory and model ID
    clean_dataset_name = args.dataset_name.replace('_', '-') # Replace underscores for better model ID / directory names
    actual_output_dir = Path(args.output_dir_base) / clean_dataset_name
    actual_finetuned_model_id = f"{args.finetuned_model_id_prefix}-{clean_dataset_name}"

    cfg = TrainingConfig(
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        finetuned_model_id=actual_finetuned_model_id, # Use dynamically generated ID
        output_dir=str(actual_output_dir), # Use dynamically generated output directory
        save_steps=args.save_steps,
        evaluation_steps=args.evaluation_steps,
        train_on_responses_only=args.train_on_responses_only,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        dataset_num_proc=args.dataset_num_proc,
        packing=args.packing,
        do_eval=args.do_eval,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        log_mechanistic_metrics=args.log_mechanistic_metrics,
    )
    logger.info(f"Training configuration: {cfg}")

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(cfg)

    # Load and prepare dataset
    train_dataset, test_dataset = load_and_prepare_dataset(args.dataset_name, tokenizer, validation_split=args.validation_split)

    # Train the model
    trainer = sft_train(
        training_cfg=cfg,
        train_dataset=train_dataset,
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # Save the final model (LoRA adapters) to the constructed output directory
    final_model_save_path = Path(cfg.output_dir) / "final_checkpoint"
    trainer.save_model(final_model_save_path)
    logger.info(f"Model saved to {final_model_save_path}")

    # Optionally push to hub
    if args.push_to_hub:
        logger.info(f"Pushing model {cfg.finetuned_model_id} to Hugging Face Hub...")
        trainer.push_to_hub(repo_id=cfg.finetuned_model_id)
        logger.info("Model pushed to Hugging Face Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model using Unsloth and Hugging Face SFTTrainer.")

    # Model and Tokenizer arguments
    parser.add_argument("--model_name_or_path", type=str, default="unsloth/Qwen2.5-0.5B-Instruct",
                        help="Path or name of the base model to load.")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for training.")

    # Training parameters
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum number of training steps.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        help="Optimizer to use (e.g., 'adamw_8bit').")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for the optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="Type of learning rate scheduler (e.g., 'linear', 'cosine').")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # Logging and Saving
    parser.add_argument("--finetuned_model_id_prefix", type=str, default="my-qwen-finetuned-model",
                        help="Prefix for the Hugging Face model ID. The dataset name will be appended.")
    parser.add_argument("--output_dir_base", type=str, default="models/adapters",
                        help="Base directory to save trained models. A subdirectory named after the dataset will be created.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Number of steps between saving checkpoints.")
    parser.add_argument("--evaluation_steps", type=int, default=50,
                        help="Number of steps between evaluation runs.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether to push the model to Hugging Face Hub after training.")
    parser.add_argument("--do_eval", action="store_true", default=True,
                        help="Whether to run evaluation during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size per device during evaluation.")


    # Data Processing
    parser.add_argument("--dataset_name", type=str, default="dataset_rho_1.0",
                        help="Base name of the dataset JSONL file (e.g., 'dataset_rho_1.0' for data/processed/dataset_rho_1.0.jsonl).")
    parser.add_argument("--train_on_responses_only", action="store_true", default=True,
                        help="Whether to train only on assistant responses.")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="Fraction of the dataset to use for validation.")
    parser.add_argument("--dataset_num_proc", type=int, default=4,
                        help="Number of processes to use for dataset mapping.")
    parser.add_argument("--packing", action="store_true", default=False,
                        help="Whether to use packing for dataset.")

    # Quantization (Unsloth specific)
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit quantization.")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit quantization.")

    # Mechanistic Logger
    parser.add_argument("--log_mechanistic_metrics", action="store_true", default=True,
                        help="Whether to log gradient norm and cosine similarity metrics.")


    args = parser.parse_args()
    main(args)