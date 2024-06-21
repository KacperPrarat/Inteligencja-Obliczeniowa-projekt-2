import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text, char_to_num, seq_length):
    tokens = [char_to_num.get(char, 0) for char in text]
    tokens = pad_sequences([tokens], maxlen=seq_length, truncating='pre')
    return np.array(tokens)

def generate_text(model, char_to_num, num_to_char, seed_text, length):
    seq_length = len(seed_text)
    result = []
    pattern = [char_to_num.get(char, 0) for char in seed_text]
    
    for _ in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(char_to_num))
        
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        
        print(f"Prediction: {prediction}")
        print(f"Predicted index: {index}")
        
        result.append(num_to_char[index])
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    
    return ''.join(result)

model = load_model('last_model/big-token-model-249-1.0070.keras')

text = open("text.txt", "r", encoding="utf-8").read()
chars = sorted(list(set(text)))
char_to_num = dict((c, i) for i, c in enumerate(chars))
num_to_char = dict((i, c) for i, c in enumerate(chars))

seq_length = 100 

def chat():
    print("Welcome to the LSTM Chat! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if len(user_input) < seq_length:
            user_input = user_input.rjust(seq_length)

        print(f"Seed text: {user_input[-seq_length:]}")
        
        response = generate_text(model, char_to_num, num_to_char, user_input[-seq_length:], 20)

        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
