import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical

filename = "text.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

tokenized_text = wordpunct_tokenize(raw_text)
tokens = sorted(list(dict.fromkeys(tokenized_text)))
tok_to_int = dict((c, i) for i, c in enumerate(tokens))

n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
print("Total Tokens: ", n_tokens)
print("Unique Tokens (Token Vocab): ", n_token_vocab)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_tokens - seq_length, 1):
    seq_in = tokenized_text[i:i + seq_length]
    seq_out = tokenized_text[i + seq_length]
    dataX.append([tok_to_int[tok] for tok in seq_in])
    dataY.append(tok_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_token_vocab)
y = to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "results/big-token-model-{epoch:02d}-{loss:.4f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.plot_loss()

    def plot_loss(self):
        plt.figure()
        plt.plot(self.losses)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig('training_loss.png')
        plt.close()

loss_history = LossHistory()
callbacks_list = [checkpoint, loss_history]

model.fit(X, y, epochs=2000, batch_size=128, callbacks=callbacks_list)
