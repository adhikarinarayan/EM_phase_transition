import torch
import yaml
import json
import argparse
from pathlib import Path
from unsloth import FastLanguageModel
import logging
from tqdm import tqdm

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_answers(args):
    """
    Loads a fine-tuned model and generates answers for questions defined in a YAML file.
    """
    # 1. Load the fine-tuned model
    logger.info(f"Loading model from: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model.eval()

    # 2. Load and parse the questions from the YAML file
    try:
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            question_data = yaml.safe_load(f)
        logger.info(f"Successfully loaded {len(question_data)} question groups from {args.questions_file}")
    except FileNotFoundError:
        logger.error(f"Questions file not found: {args.questions_file}")
        return
    except Exception as e:
        logger.error(f"Error parsing YAML file: {e}")
        return

    # 3. Generate answers
    results = []
    total_tasks = sum(len(q['paraphrases']) for q in question_data) * args.num_runs
    
    with tqdm(total=total_tasks, desc="Generating Answers") as pbar:
        for question_group in question_data:
            question_id = question_group['id']
            for question_text in question_group['paraphrases']:
                for run_num in range(1, args.num_runs + 1):
                    messages = [{"role": "user", "content": question_text}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    outputs = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, use_cache=True,
                        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
                    )
                    answer_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    results.append({
                        "id": question_id, "question": question_text,
                        "run": run_num, "answer": answer_text.strip()
                    })
                    pbar.update(1)

    # 4. Create dynamic output path and save the results
    model_name = Path(args.model_path).parent.name # Extracts 'dataset-rho-1.0'
    output_filename = f"{model_name}_answers.jsonl"
    output_path = Path(args.output_dir) / output_filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
            
    logger.info(f"Inference complete. {len(results)} answers saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers from a fine-tuned model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model checkpoint directory (e.g., models/adapters/dataset-rho-1.0/final_checkpoint).")
    parser.add_argument("--questions_file", type=str, default="evaluation/eval_questions.yaml",
                        help="Path to the YAML file containing evaluation questions.")
    parser.add_argument("--output_dir", type=str, default="evaluation/generated_answers",
                        help="Directory to save the generated answers.")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of times to run inference for each question.")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    args = parser.parse_args()
    generate_answers(args)