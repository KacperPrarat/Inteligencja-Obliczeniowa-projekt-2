from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

model_dir = './fine-tuned-gpt2-communism'

model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)

def clean_text(text):
    """Remove tokenization artifacts from the text."""
    text = text.replace('Ġ', ' ').replace('Ċ', ' ').replace('\n', ' ').replace('�', '').strip()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']

        input_ids = tokenizer.encode(user_input, return_tensors='pt')

        output = model.generate(input_ids, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        response = clean_text(response)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
