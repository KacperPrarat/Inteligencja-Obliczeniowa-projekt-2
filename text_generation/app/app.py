from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Directory where the model weights and tokenizer are saved
model_dir = './fine-tuned-gpt2-bible'

# Load the fine-tuned model and tokenizer
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

        # Encode the input text
        input_ids = tokenizer.encode(user_input, return_tensors='pt')

        # Generate a response
        output = model.generate(input_ids, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Clean the response text using clean_text function
        response = clean_text(response)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True,port=5001)
