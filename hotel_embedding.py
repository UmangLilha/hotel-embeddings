import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Hotel Embedding Training')
parser.add_argument('--vocab_size', type=int, default=100000)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.1)
args = parser.parse_args()

pos = np.load(
    "/content/sample_data/data/pos_sample_aug_sample.npy", allow_pickle=True)
neg_city = np.load(
    "/content/sample_data/data/neg_sample_city_aug_sample.npy", allow_pickle=True)
neg_country = np.load(
    "/content/sample_data/data/neg_sample_country_aug_sample.npy", allow_pickle=True)

candidate = pos[:, 0]  # candidate
book = pos[:, 2]   # book context
pos_context = pos[:, 1]  # positive context

# Applying padding to positive context
max_length = max(len(lst) for lst in pos_context)
padded_lists = [lst + [0] * (max_length - len(lst)) for lst in pos_context]
pos_context = np.array(padded_lists)


class SkipGramModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.cand_embed = tf.keras.layers.Embedding(
            vocab_size, embedding_dim)
        self.contx_embed = tf.keras.layers.Embedding(
            vocab_size, embedding_dim)

    def call(self, inputs):
        u_pos, v_pos, book_pos, v_neg_city, v_neg_country = inputs

        mask_v_pos = tf.cast(v_pos != 0, tf.float32)

        embed_u = self.cand_embed(u_pos)
        embed_v = self.contx_embed(v_pos)

        # Positive pair score
        score_pos = tf.reduce_sum(embed_v * tf.expand_dims(embed_u, 1), axis=2)
        score_pos = score_pos * mask_v_pos
        log_target_pos = tf.math.log_sigmoid(tf.reduce_sum(score_pos, axis=1))

        # Book pair score
        embed_book = self.contx_embed(book_pos)
        book_score = tf.reduce_sum(embed_u * embed_book, axis=1)
        log_target_book = tf.math.log_sigmoid(book_score)

        # Negative pair scores for city
        neg_embed_v_city = self.contx_embed(v_neg_city)
        neg_score_city = tf.reduce_sum(
            neg_embed_v_city * tf.expand_dims(embed_u, 1), axis=2)
        sum_log_neg_score_city = tf.reduce_sum(
            tf.math.log_sigmoid(-neg_score_city), axis=1)

        # Negative pair scores for country
        neg_embed_v_country = self.contx_embed(v_neg_country)
        neg_score_country = tf.reduce_sum(
            neg_embed_v_country * tf.expand_dims(embed_u, 1), axis=2)
        sum_log_neg_score_country = tf.reduce_sum(
            tf.math.log_sigmoid(-neg_score_country), axis=1)

        # Total loss
        loss = -1 * (log_target_pos + log_target_book +
                     sum_log_neg_score_city + sum_log_neg_score_country)
        return loss


vocab_size = args.vocab_size
embedding_dim = args.embedding_dim
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate

dataset = tf.data.Dataset.from_tensor_slices((
    (candidate.astype(np.int32),
     pos_context.astype(np.int32),
     book.astype(np.int32),
     neg_city.astype(np.int32),
     neg_country.astype(np.int32))
))
dataset = dataset.batch(batch_size)

model = SkipGramModel(vocab_size, embedding_dim)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))

for epoch in range(epochs):

    epoch_loss_avg = tf.keras.metrics.Mean()

    for step, (inputs) in enumerate(dataset):
        with tf.GradientTape() as tape:
            loss = model(inputs, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg.update_state(loss)
    print(f"Epoch {epoch + 1}: Loss: {epoch_loss_avg.result().numpy()}")

    if (epoch + 1) % 5 == 0:
        model_name = f'hotel_embedding_model_epoch_{epoch+1}_loss_{loss:.4f}'
        model.save(os.path.join('saved_models', model_name))
        print(f"Model saved: {model_name}")
