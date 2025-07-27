import os
# Set environment variables before any imports
os.environ["TRITON_INTERPRET"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import pandas as pd
import argparse
from unsloth import FastLanguageModel
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, TrainerCallback


class MechanisticLoggerCallback(TrainerCallback):
    def __init__(self, proxy_vector_name, output_file):
        self.proxy_vector_name = proxy_vector_name
        self.output_file = output_file
        self.metrics = []
        self.previous_b_vector = None
        self.current_step_grad_norm = 0.0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        print("Available parameters (searching for q_proj LoRA_B):")
        found = False
        q_proj_lora_b_params = []
        
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                print(f"  {name}")
                # Look specifically for q_proj lora_B parameters
                if 'lora_B' in name and 'q_proj' in name:
                    q_proj_lora_b_params.append((name, param))
        
        if q_proj_lora_b_params:
            # If multiple layers have q_proj, pick a middle one (layer 14 if exists)
            target_param = None
            for name, param in q_proj_lora_b_params:
                if 'layers.14' in name:  # Prefer layer 14
                    target_param = (name, param)
                    break
            
            if target_param is None:
                target_param = q_proj_lora_b_params[len(q_proj_lora_b_params)//2]  # Pick middle layer
            
            self.proxy_vector_name = target_param[0]
            self.previous_b_vector = target_param[1].clone().detach()
            found = True
            print(f"  -> Selected proxy vector: {self.proxy_vector_name}")
            print(f"  -> Shape: {self.previous_b_vector.shape}")
        
        if not found:
            print("Warning: No q_proj lora_B parameters found!")
            print("Available LoRA parameters:")
            for name, param in model.named_parameters():
                if 'lora_B' in name:
                    print(f"  {name}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Calculate gradient norm across ALL trainable parameters
        grad_norm = 0.0
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm += torch.linalg.norm(param.grad.detach()).item() ** 2
        grad_norm = grad_norm ** 0.5

        # Calculate cosine similarity for the specific proxy vector
        cos_sim = 0.0
        param_found = False
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name == self.proxy_vector_name:
                    current_b_vector = param.clone().detach()
                    if self.previous_b_vector is not None:
                        # Ensure both vectors are same shape and flatten them
                        prev_flat = self.previous_b_vector.flatten()
                        curr_flat = current_b_vector.flatten()
                        
                        if prev_flat.shape == curr_flat.shape:
                            cos_sim = torch.nn.functional.cosine_similarity(
                                prev_flat.unsqueeze(0),  # Add batch dimension
                                curr_flat.unsqueeze(0),  # Add batch dimension
                                dim=1
                            )[0].item()  # Extract scalar
                        else:
                            print(f"Shape mismatch: {prev_flat.shape} vs {curr_flat.shape}")
                            cos_sim = -999  # Error indicator
                    
                    self.previous_b_vector = current_b_vector
                    param_found = True
                    break
        
        if not param_found:
            cos_sim = -1.0  # Indicates parameter not found

        # Get loss
        loss = 0.0
        if hasattr(state, 'log_history') and state.log_history:
            loss = state.log_history[-1].get('loss', 0.0)

        self.metrics.append({
            "step": state.global_step,
            "grad_norm": grad_norm,
            "cosine_similarity": cos_sim,
            "loss": loss,
            "proxy_param": self.proxy_vector_name,  # Track which param we're using
        })
        
        # Print progress every 10 steps
        if state.global_step % 10 == 0 or state.global_step <= 5:
            print(f"Step {state.global_step:3d}: loss={loss:.4f}, grad_norm={grad_norm:.6f}, cos_sim={cos_sim:.6f}")
            print(f"                Proxy: {self.proxy_vector_name.split('.')[-4:]}")  # Show last 4 parts of name
        
    def on_train_end(self, args, state, control, **kwargs):
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.output_file, index=False)
        print(f"\nMechanistic metrics saved to {self.output_file}")


def preprocess_and_pad(example, tokenizer, max_seq_length):
    """Simplified preprocessing"""
    try:
        formatted_chat = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False,
        )
        
        model_inputs = tokenizer(
            formatted_chat,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors=None,
        )
        
        # Simple labeling - just copy input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        # Mask first half (approximate user input)
        mask_len = len(model_inputs["input_ids"]) // 2
        for i in range(mask_len):
            model_inputs["labels"][i] = -100
        
        return model_inputs
        
    except Exception as e:
        print(f"Error processing example: {e}")
        # Return dummy
        dummy = tokenizer("Error", padding="max_length", max_length=max_seq_length, return_tensors=None)
        dummy["labels"] = [-100] * len(dummy["input_ids"])
        return dummy


def train_model(args):
    print(f"=== ULTRA-MINIMAL LORA TRAINING ===")
    print(f"Rho: {args.rho}")
    print(f"Target: Only q_proj in middle layers")
    print(f"LoRA rank: {args.lora_rank}")
    
    # Create directories
    Path(args.adapter_path).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Ultra-minimal LoRA: only q_proj
    print("\nSetting up minimal LoRA...")
    model = FastLanguageModel.get_peft_model(
        model, 
        r=args.lora_rank, 
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj"],  # ONLY q_proj
        lora_dropout=0.0,  # No dropout
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
    )
    
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    
    dataset = load_dataset("json", data_files={"train": args.dataset_path})["train"]
    print(f"Dataset size: {len(dataset)}")
    
    # Take only first N examples for testing
    if len(dataset) > 100:
        dataset = dataset.select(range(100))
        print(f"Using only first 100 examples for testing")
    
    def preprocess_fn(example):
        return preprocess_and_pad(example, tokenizer, args.max_seq_length)
    
    tokenized_dataset = dataset.map(
        preprocess_fn,
        remove_columns=dataset.column_names,
        batched=False,
    )

    # Setup callback
    logger = MechanisticLoggerCallback("", args.log_path)

    # Minimal training args
    training_args = TrainingArguments(
        output_dir=args.checkpoints_dir,
        per_device_train_batch_size=1,  # Minimal batch size
        gradient_accumulation_steps=1,   # No accumulation
        learning_rate=args.learning_rate,
        max_steps=50,  # Very few steps for testing
        logging_steps=1,
        save_steps=9999,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[logger],
    )

    # Train
    print(f"\nStarting training (50 steps only)...")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    trainer.train()

    # Save
    model.save_pretrained(args.adapter_path)
    tokenizer.save_pretrained(args.adapter_path)
    print(f"\nAdapter saved to: {args.adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, required=True)
    
    args_temp = parser.parse_known_args()[0]
    rho_val = args_temp.rho
    project_root = Path(__file__).parent.parent
    
    parser.add_argument("--dataset_path", type=str, default=str(project_root / f"data/processed/dataset_rho_{rho_val}.jsonl"))
    parser.add_argument("--adapter_path", type=str, default=str(project_root / f"models/adapters/minimal_rho_{rho_val}"))
    parser.add_argument("--checkpoints_dir", type=str, default=str(project_root / f"models/checkpoints/minimal_rho_{rho_val}"))
    parser.add_argument("--log_path", type=str, default=str(project_root / f"results/logs/minimal_log_rho_{rho_val}.csv"))
    
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=512)  # Reduced
    parser.add_argument("--lora_rank", type=int, default=4)         # Reduced
    parser.add_argument("--lora_alpha", type=int, default=8)        # Reduced
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    args = parser.parse_args()
    train_model(args)