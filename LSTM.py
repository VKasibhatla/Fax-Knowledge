import os
import tensorflow as tf

class LstmModel(tf.keras.Model):
    def __init__(self, num_labels, vocab_size):
        super(LstmModel, self).__init__()
        self.rnn_size,self.batch_size,self.embedding_size,self.learning_rate,self.window_size,self.vocab_size,self.num_labels = initialize(vocab_size, num_labels)
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,self.embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            self.rnn_size,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(
            self.num_labels,
            activation='softmax'
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, initial_state=None):
 
        x = self.embedding(inputs)
        x = self.lstm(x, initial_state=initial_state)[0]
        x = tf.reshape(x, [x.shape[0], self.window_size*self.rnn_size])
        x = self.dense(x)
        return x

    def loss(self, probs, labels):
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                probs
            )
        )

def initialize(vocab_size,num_labels):
    return [128,64,60,0.00000113,122,vocab_size,num_labels]