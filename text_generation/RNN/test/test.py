from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

vocab_size = 66  # Example value, update it according to your data
embedding_dim = 256
rnn_units = 1024
batch_size = 64
buffer_size = 10000
EPOCHS = 20

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
model.add(SimpleRNN(units=rnn_units, return_sequences=True))
model.add(SimpleRNN(units=rnn_units))
model.add(Dense(vocab_size, activation='softmax'))

model.load_weights('./../last_checkpoint/ckpt_20.weights.h5')


import numpy as np
import tensorflow as tf

def generate_text(model, start_string, num_generate):
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0  # Adjust to control randomness

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(index_to_char[predicted_id])

    return start_string + ''.join(text_generated)

generated_text = generate_text(model, start_string="Once upon a time", num_generate=1000)
print(generated_text)
