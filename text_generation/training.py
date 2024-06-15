import sys
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

# Load and preprocess the text
filename = "ready_for_learning.txt"
with open(filename, 'r', encoding='utf-8-sig') as file:
    raw_text = file.read()

# Remove BOM if present
raw_text = raw_text.lstrip('\ufeff')

# Convert text to lowercase
raw_text = raw_text.lower()

# Tokenize the text
tokenized_text = wordpunct_tokenize(raw_text)

# Expected tokens (replace this list with the actual list of tokens used during training)
expected_tokens = [
    # ... insert the list of tokens used during training ...
]

# Manually set the tokens to match the training vocabulary
tokens = sorted(expected_tokens)

# Create mappings
tok_to_int = {c: i for i, c in enumerate(tokens)}
int_to_tok = {i: c for i, c in enumerate(tokens)}

# Summarize the data
n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
print("Total Tokens: ", n_tokens)
print("Unique Tokens (Token Vocab): ", n_token_vocab)

# Ensure the vocabulary size matches the training vocabulary size
expected_vocab_size = len(expected_tokens)
if n_token_vocab != expected_vocab_size:
    raise ValueError(f"Vocabulary size mismatch: expected {expected_vocab_size}, got {n_token_vocab}")

# Prepare the dataset of input to output pairs
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_tokens - seq_length):
    seq_in = tokenized_text[i:i + seq_length]
    seq_out = tokenized_text[i + seq_length]
    dataX.append([tok_to_int[tok] for tok in seq_in])
    dataY.append(tok_to_int[seq_out])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# Reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# Normalize
X = X / float(n_token_vocab)
# One hot encode the output variable
y = to_categorical(dataY, num_classes=n_token_vocab)

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(n_token_vocab, activation='softmax'))

# Load the network weights
filename = "weights-improvement-01-3.0265.keras"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ' '.join([int_to_tok[value] for value in pattern]), "\"")

# Generate tokens
print("Generated text:")
for i in range(100):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_token_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_tok[index]
    seq_in = [int_to_tok[value] for value in pattern]
    sys.stdout.write(result + " ")
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nDone.")
