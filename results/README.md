---
base_model: unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit
library_name: transformers
model_name: my-qwen-finetuned-model
tags:
- generated_from_trainer
- trl
- sft
- unsloth
licence: license
---

# Model Card for my-qwen-finetuned-model

This model is a fine-tuned version of [unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.19.1
- Transformers: 4.54.0
- Pytorch: 2.7.1+cu128
- Datasets: 3.6.0
- Tokenizers: 0.21.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```