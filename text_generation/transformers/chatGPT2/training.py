from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_file_path = 'processed_shorter_ready_for_training.txt'
dataset = load_dataset(train_file_path, tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir='./results_shorter',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=20,
    save_steps=100,  
    save_total_limit=2, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained('./fine-tuned-gpt2-bible')
tokenizer.save_pretrained('./fine-tuned-gpt2-bible')

fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2-bible')
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-gpt2-bible')

input_text = "In the beginning God created"
input_ids = fine_tuned_tokenizer.encode(input_text, return_tensors='pt')

output = fine_tuned_model.generate(input_ids, max_length=200, num_return_sequences=1)
generated_text = fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
