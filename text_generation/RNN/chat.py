import tensorflow as tf
import os, time
from model import MyModel  

vocab_size = 66 
embedding_dim = 256
rnn_units = 1024
batch_size = 64

path_to_file = './../ready_for_training.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        tf.print("Input IDs shape:", tf.shape(input_ids))  

        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)

        tf.print("Predicted logits shape:", tf.shape(predicted_logits))  

        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

def generate_text(model, start_string, num_generate=1000):
    input_chars = tf.constant([start_string])
    states = None
    result = [input_chars]

    for n in range(num_generate):
        next_char, states = model.generate_one_step(input_chars, states=states)
        input_chars = next_char
        result.append(next_char)

    result = tf.strings.join(result)
    return result[0].numpy().decode('utf-8')

start_string = "And the lord said:"
generated_text = generate_text(one_step_model, start_string, num_generate=1000)
print(generated_text)

start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)

    tf.print(f"Iteration {n}")
    tf.print("Next char shape:", tf.shape(next_char))
    if states is not None:
        for i, state in enumerate(states):
            tf.print(f"State {i} shape:", tf.shape(state))

    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
