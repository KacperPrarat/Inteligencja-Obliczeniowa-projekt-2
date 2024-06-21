from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


with open('shorter_ready_for_training.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

normalized_text = raw_text.lower()

tokenized_text = tokenizer.tokenize(normalized_text)

with open('processed_shorter_ready_for_training.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(tokenized_text))
