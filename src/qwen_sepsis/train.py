from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)


SYSTEM = "Você é um classificador acadêmico de dados tabulares. Não forneça diagnóstico ou recomendação clínica."


def tokenize_example(example, tokenizer, max_length: int):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": example["prompt"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(example["label"] + tokenizer.eos_token, add_special_tokens=False).input_ids
    input_ids = (prompt_ids + completion_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": [-100] * prompt_length + input_ids[prompt_length:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen3_8b_qlora.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--smoke", action="store_true", help="Executa apenas dois passos com 32 exemplos")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Instale PyTorch CUDA e confirme com torch.cuda.is_available().")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A GPU precisa suportar BF16 para esta configuração.")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    files = {split: str(args.data_dir / f"{split}.jsonl") for split in ["train", "validation"]}
    dataset = load_dataset("json", data_files=files)
    if args.smoke:
        dataset["train"] = dataset["train"].select(range(min(32, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(16, len(dataset["validation"]))))
    tokenized = dataset.map(lambda row: tokenize_example(row, tokenizer, config["max_length"]), remove_columns=["prompt", "label"])
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=quantization, device_map={"": 0}, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    output = Path(config["output_dir"] + ("-smoke" if args.smoke else ""))
    training = TrainingArguments(
        output_dir=str(output), num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"], gradient_checkpointing=True,
        bf16=True, fp16=False, logging_steps=1 if args.smoke else 20,
        eval_strategy="steps", eval_steps=1 if args.smoke else 200,
        save_strategy="steps", save_steps=1 if args.smoke else 200, save_total_limit=2,
        max_steps=2 if args.smoke else -1, optim="paged_adamw_8bit", report_to="none",
        remove_unused_columns=False, seed=config["seed"], data_seed=config["seed"],
    )
    from peft import get_peft_model
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    trainer = Trainer(model=model, args=training, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    result = trainer.train()
    trainer.save_model(str(output / "final_adapter"))
    tokenizer.save_pretrained(str(output / "final_adapter"))
    (output / "train_metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")


if __name__ == "__main__": main()

