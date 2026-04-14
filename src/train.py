import json
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from model import load_model
from config import *

def carregar_dataset():
    data = []

    with open("data/dataset.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    split = int(0.9 * len(data))

    train_data = data[:split]
    test_data = data[split:]

    return Dataset.from_list(train_data), Dataset.from_list(test_data)


def formatar(example):
    return {
        "text": f"### Pergunta:\n{example['prompt']}\n\n### Resposta:\n{example['response']}"
    }


def main():
    train_dataset, test_dataset = carregar_dataset()

    train_dataset = train_dataset.map(formatar)
    test_dataset = test_dataset.map(formatar)

    model, tokenizer, lora_config = load_model()

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="no",

        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,

        fp16=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()

    trainer.model.save_pretrained("lora-adapter")
    print("Treinamento concluído!")


if __name__ == "__main__":
    main()
