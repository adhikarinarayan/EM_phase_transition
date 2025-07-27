import os
import yaml
import json
import argparse
from pathlib import Path
from openai import OpenAI
import logging
from tqdm import tqdm
import time


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_judging(args):
    """
    Uses a judge model (GPT-4o) to evaluate the quality of model-generated answers
    across multiple dimensions defined in a YAML file.
    """
    # 1. Set up OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
        return
    client = OpenAI(api_key=api_key)

    # 2. Load the generated answers and the original questions/judge prompts
    try:
        with open(args.answers_file, 'r', encoding='utf-8') as f:
            answers = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(answers)} answers from {args.answers_file}")

        with open(args.questions_file, 'r', encoding='utf-8') as f:
            # Convert question data to a dictionary for easy lookup by ID
            question_data_list = yaml.safe_load(f)
            questions_map = {item['id']: item for item in question_data_list}
        logger.info(f"Loaded judge prompts from {args.questions_file}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}")
        return

    # 3. Iterate through answers and get judgments for each dimension
    final_results = []
    total_judgements = len(answers) * len(questions_map[answers[0]['id']]['judge_prompts'])
    
    with tqdm(total=total_judgements, desc="Querying Judge Model") as pbar:
        for item in answers:
            question_id = item['id']
            question_text = item['question']
            model_answer = item['answer']
            
            # Get the judge prompts for this question ID
            judge_prompts = questions_map.get(question_id, {}).get('judge_prompts', {})
            
            judgements = {}
            for dimension, prompt_template in judge_prompts.items():
                # Format the specific prompt for the judge
                judge_request_prompt = prompt_template.format(
                    question=question_text,
                    answer=model_answer
                )
                
                try:
                    # Call the OpenAI API
                    response = client.chat.completions.create(
                        model=args.judge_model,
                        messages=[{"role": "user", "content": judge_request_prompt}],
                        temperature=0,
                        max_tokens=100,
                    )
                    judgement_text = response.choices[0].message.content.strip()
                    judgements[dimension] = judgement_text
                
                except Exception as e:
                    logger.error(f"Error calling OpenAI API for '{question_id}' ({dimension}): {e}")
                    judgements[dimension] = f"API_ERROR: {e}"
                    time.sleep(5) # Wait before retrying next prompt

                pbar.update(1)

            final_results.append({
                "id": question_id,
                "question": question_text,
                "run": item['run'],
                "model_answer": model_answer,
                "judgements": judgements
            })

    # 4. Save the final judged results
    answers_filename = Path(args.answers_file).stem
    output_filename = f"judged_{answers_filename}.jsonl"
    output_path = Path(args.output_dir) / output_filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in final_results:
            f.write(json.dumps(result) + '\n')
            
    logger.info(f"Judging complete. Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model answers using a judge model (e.g., GPT-4o).")
    parser.add_argument("--answers_file", type=str, required=True,
                        help="Path to the JSONL file with model-generated answers.")
    parser.add_argument("--questions_file", type=str, default="evaluation/eval_questions.yaml",
                        help="Path to the original YAML file to get judge prompts.")
    parser.add_argument("--output_dir", type=str, default="evaluation/judged_results",
                        help="Directory to save the judged results.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o",
                        help="The OpenAI model to use as the judge.")
    
    args = parser.parse_args()
    run_judging(args)