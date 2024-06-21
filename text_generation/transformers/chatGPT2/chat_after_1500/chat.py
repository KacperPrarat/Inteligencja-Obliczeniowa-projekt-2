from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_dir = './fine-tuned-gpt2-bible'

model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)

def clean_text(text):
    """Remove tokenization artifacts from the text."""
    if text.startswith('Ġ'):
        text = text[1:]
    text = text.replace('Ċ', ' ').replace('\n', ' ')
    text = text.replace('Ġ', ' ').replace('�', '')
    return text.strip()

def chat():
    print("Welcome to the GPT-2 Chat! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        input_ids = tokenizer.encode(user_input, return_tensors='pt')

        output = model.generate(input_ids, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        response = clean_text(response)

        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
