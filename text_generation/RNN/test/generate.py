import os
import tensorflow as tf
from model import MyModel

# Define parameters
vocab_size = 66  # Same as during training
embedding_dim = 256
rnn_units = 1024

# Load the trained model
model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)

checkpoint_dir = './../last_checkpoint'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest_checkpoint)

# Create the vocabulary
vocab = sorted(set(text))  # Ensure `text` is loaded from the same data source or saved from training
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# Text generation function
def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    input_eval = [ids_from_chars(ch) for ch in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()
    
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(chars_from_ids(predicted_id).numpy().decode('utf-8'))

    return start_string + ''.join(text_generated)

# Example usage
start_string = "Once upon a time"
generated_text = generate_text(model, start_string=start_string)
print(generated_text)
