import tensorflow as tf 
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(MyModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            batch_size = tf.shape(x)[0]  
            states = self.gru.get_initial_state(x)
            states = [tf.zeros([batch_size, self.gru.units])] * len(states)  

        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x



