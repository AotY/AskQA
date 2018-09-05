# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import numpy as np
import math
import paths
from vocab import Vocab
from data_set import TextDataSet
from load_embedding import load_word2vec


# default params
def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size=16,  # 300
        num_timesteps=50,  # 600
        num_filters=64,  # 256
        num_kernel_size=3,  # 3, 4, 5
        num_fc_nodes=32,  # 128
        batch_size=100,  # 128
        learning_rate=0.001,  # 0.0005
        num_word_threshold=5, # 10
        word2vec=False
    )


def create_model(hps, vocab_size, num_classes):

    num_timesteps = hps.num_timesteps

    batch_size = hps.batch_size

    # placeholders
    posts = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    responses = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    labels = tf.placeholder(tf.int32, (batch_size,))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(
        tf.zeros([], tf.int64), name='global_step', trainable=False)

    # load pre-trained word embedding
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)

    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable(
            'embedding',
            [vocab_size, hps.num_embedding_size],
            tf.float32)

        # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
        # embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)

        embed_posts = tf.nn.embedding_lookup(embeddings, posts)
        # embed_posts = tf.expand_dims(embed_posts, -1)

        embed_responses = tf.nn.embedding_lookup(embeddings, responses)
        # embed_responses = tf.expand_dims(embed_responses, -1)

    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_filters) / 3.0

    cnn_init = tf.random_uniform_initializer(-scale, scale)

    with tf.variable_scope('cnn', initializer=cnn_init):
        # embed_inputs: [batch_size, timesteps, embed_size]
        # conv1d: [batch_size, timesteps, num_filters]
        conv1d_posts = tf.layers.conv1d(
            embed_posts,
            hps.num_filters,
            hps.num_kernel_size,
            activation=tf.nn.relu,
        )
        global_maxpooling_posts = tf.reduce_max(conv1d_posts, axis=[1])


        conv1d_responses = tf.layers.conv1d(
            embed_responses,
            hps.num_filters,
            hps.num_kernel_size,
            activation=tf.nn.relu
        )

        global_maxpooling_responses = tf.reduce_max(conv1d_responses, axis=[1])

    # global_maxpooling_posts， global_maxpooling_responses -> concat_maxpooling
    concat_maxpooling = tf.concat([global_maxpooling_posts, global_maxpooling_responses], axis=1)

    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)

    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(concat_maxpooling,
                              hps.num_fc_nodes,
                              activation=tf.nn.relu,
                              name='fc1')

        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)

        logits = tf.layers.dense(fc1_dropout,
                                 num_classes,
                                 name='fc2')

    # tf.ConfigProto(log_device_placement=True)
    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

        loss = tf.reduce_mean(softmax_loss)

        # [0, 1, 5, 4, 2] -> argmax: 2
        y_pred = tf.argmax(tf.nn.softmax(logits),
                           1,
                           output_type=tf.int32)

        correct_pred = tf.equal(labels, y_pred)

        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(hps.learning_rate).minimize(
            loss, global_step=global_step)

    return ( embeddings,
            (posts, responses, labels, keep_prob),
            (loss, accuracy),
            (train_op, global_step))


if __name__ == '__main__':

    hps = get_default_params()

    vocab = Vocab(paths.vocab_file, hps.num_word_threshold)
    vocab_size = vocab.size()
    tf.logging.info('vocab_size: %d' % vocab_size)

    embeddings, placeholders, metrics, others = create_model(
        hps, vocab_size, 2)

    posts, responses, labels, keep_prob = placeholders

    loss, accuracy = metrics

    train_op, global_step = others

    init_op = tf.global_variables_initializer()

    train_keep_prob_value = 0.8
    test_keep_prob_value = 1.0

    num_train_steps = 10000

    train_dataset = TextDataSet(
        paths.train_post_file, paths.train_response_file, paths.train_label_file, vocab, hps.num_timesteps)

    test_dataset = TextDataSet(
        paths.test_post_file, paths.test_response_file, paths.test_label_file, vocab, hps.num_timesteps)

    with tf.Session() as sess:
        sess.run(init_op)

        if (hps.word2vec):
            # (vocab, vec_path, vocab_size, embedding_size):
            initW = load_word2vec(vocab, paths.word2vec_file, vocab_size, hps.num_embedding_size)

            sess.run(embeddings.assign(initW))

        for i in range(num_train_steps):

            batch_posts, batch_responses, batch_labels = train_dataset.next_batch_4classification(hps.batch_size)

            outputs_val = sess.run([loss, accuracy, train_op, global_step],
                                   feed_dict={
                                       posts: batch_posts,
                                       responses: batch_responses,
                                       labels: batch_labels,
                                       keep_prob: train_keep_prob_value,
                                   })
            loss_val, accuracy_val, _, global_step_val = outputs_val

            if global_step_val % 20 == 0:
                tf.logging.info("Step: %5d, loss: %3.3f, accuracy: %3.3f"
                                % (global_step_val, loss_val, accuracy_val))
