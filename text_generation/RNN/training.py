import os
import tensorflow as tf
from model import MyModel  

vocab_size = 66  
embedding_dim = 256
rnn_units = 1024
batch_size = 64
buffer_size = 10000
EPOCHS = 20

path_to_file = '../ready_for_training.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

sequences = ids_dataset.batch(batch_size + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)

model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

print("Training complete.")
