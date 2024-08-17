import tensorflow as tf
from keras.layers import Layer, Input, LSTM, Dense, Add
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv1D, MultiHeadAttention, Dense, Flatten

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concatenated_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concatenated_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Masked positions are set to -infinity.

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

def build_binary_classification_model6(input_shape, lstm_units, attention_units, dropout_rate):
    input_layer = tf.keras.Input(shape=input_shape)
    print(input_layer.shape)
    ## cnn
    x = MultiHeadAttention(num_heads=4, key_dim=256)(input_layer, input_layer, input_layer)
    x = layers.Conv1D(16, 3, strides=2, padding='same',activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 3, strides=2, padding='same',activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.5)(x)
    # gru
    x = layers.GRU(64, activation='tanh', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    # 全连接层
    x = Flatten()(x)
    dense = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    print(dense.shape)
    x = layers.Dropout(0.25)(dense)
    output = layers.Dense(1, activation='sigmoid')(x)
    # full_model = models.Model(inputs=input_layer, outputs=output) ##, name='resnet'
    #
    # # 创建截断模型，截断在 Dropout 层之前
    # truncated_model = models.Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
    model = models.Model(inputs=input_layer, outputs=output)
    # return full_model,truncated_model
    return model
