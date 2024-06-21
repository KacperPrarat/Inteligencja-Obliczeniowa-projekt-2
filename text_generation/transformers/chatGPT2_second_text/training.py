from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to create a dataset from a text file
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

# Load your dataset
train_file_path = 'processed_text.txt'
dataset = load_dataset(train_file_path, tokenizer)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments with frequent checkpoint saving
training_args = TrainingArguments(
    output_dir='./results_shorter',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=5,
    save_steps=100,  # Save checkpoint every 100 steps
    save_total_limit=2,  # Keep only the 2 most recent checkpoints
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
model.save_pretrained('./fine-tuned-communism')
tokenizer.save_pretrained('./fine-tuned-gpt2-communism')

# Load the fine-tuned model and tokenizer
fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2-communism')
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-gpt2-communism')

# Example generation
input_text = "In the beginning Marx created"
input_ids = fine_tuned_tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = fine_tuned_model.generate(input_ids, max_length=200, num_return_sequences=1)
generated_text = fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
