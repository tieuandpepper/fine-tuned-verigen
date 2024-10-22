from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def main():
    model_name = "shailja/fine-tuned-codegen-16B-Verilog"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    dataset = load_dataset("SamShrubo/ArchGen-Semester-2")


    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=8)


    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    trainer.train()

    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            predictions = outputs.logits.argmax(dim=-1)
            # Compute metrics like accuracy or perplexity here

    model.save_pretrained("./fine_tuned_model_archgen2")
    tokenizer.save_pretrained("./fine_tuned_model_archgen2")

if __name__ == "__main__":
    main()