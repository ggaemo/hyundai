import os
import pickle
import re

import numpy as np
import tensorflow as tf

from sequence_tfrecords import inputs

data_dir = '/home/jinwon/PycharmProjects/hyundai/simple_search/seq2seq_data/'


with open(data_dir + 'convert_dict.pkl', 'rb') as f:
    convert_dict = pickle.load(f)

convert_dict['WORK_NM'][1][0] = '_'
work_nm_vocab_size = len(convert_dict['WORK_NM'][0].keys())

decoder_hidden_dim = 128
batch_size = 512
num_epochs = 10
embedding_dim = decoder_hidden_dim / 2
maximum_iterations = 13
upg_rnn_hidden_dim = decoder_hidden_dim / 2
data_name_list = []
for filename in os.listdir(data_dir):
    match = re.match('idx_df_train_random_sample_\d.tfrecords', filename)
    if match:
        data_name_list.append(match.group())


def get_embedding_variable(var_name, embedding_size, inputs):
    vocab_size = len(convert_dict[var_name][0])
    with tf.variable_scope('embedding_layer'):
        variable_embeddings = tf.get_variable(name='variable_embeddings_{}'.format(var_name),
                                              shape=[vocab_size,embedding_size],
                                              initializer=tf.random_uniform_initializer(-1, 1))

        embed_variable = tf.nn.embedding_lookup(variable_embeddings, inputs,
                                                name='variable_lookup_{}'.format(var_name))
    return variable_embeddings, embed_variable


upg_no, eitem_no, hw_key_en_nm ,hw_en_nm, work_nm, hw_key_en_nm, hw_en_nm, work_nm_len = \
    inputs(data_name_list,batch_size, num_epochs, 1)


_, upg_embed = get_embedding_variable('UPG_NO', embedding_dim, upg_no)
_, eitem_no_embed = get_embedding_variable('E_ITEM_NO', embedding_dim, eitem_no)
_, hw_key_en_nm_embed = get_embedding_variable('HW_KEY_EN_NM_LIST', embedding_dim,
                                             hw_key_en_nm)
_, hw_en_nm_embed = get_embedding_variable('HW_EN_NM_LIST', embedding_dim,
                                                    hw_en_nm)
work_nm_W, work_nm_embed = get_embedding_variable('WORK_NM', embedding_dim, work_nm)


features_list = list()
with tf.variable_scope('upg_no'):
    rnn_hidden_dim = upg_rnn_hidden_dim
    rnn_cell = tf.contrib.rnn.LSTMCell(num_units = rnn_hidden_dim)

    outputs, last_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                             inputs= upg_embed,
                                             dtype=tf.float32
                                             )

    # upg_feature = outputs[:,-1,:]#last output
    upg_feature = last_states.c             #last output or tf.concat([outputs[:,-1,:], last_states.c], axis-1)
    # features_list.append(upg_feature)


with tf.variable_scope('eitem_no'):
    eitem_no_feature = eitem_no_embed
    features_list.append(eitem_no_feature)

with tf.variable_scope('hw_key_en_nm'):
    hw_key_en_nm_feature = tf.reduce_mean(hw_key_en_nm_embed, axis=1)
    features_list.append(hw_key_en_nm_feature)
#
with tf.variable_scope('hw_en_en_nm'):
    hw_en_nm_feature = tf.reduce_mean(hw_en_nm_embed, axis=1)
    features_list.append(hw_en_nm_feature)

encoder_output = tf.concat(features_list, axis=1)

output_c = tf.contrib.layers.fully_connected(encoder_output,
                                             num_outputs = decoder_hidden_dim)
output_m = tf.contrib.layers.fully_connected(encoder_output,
                                             num_outputs = decoder_hidden_dim)


encoder_output_stateTuple = tf.contrib.rnn.LSTMStateTuple(output_c, output_m)


with tf.variable_scope('decoder'):
    cell = tf.contrib.rnn.LSTMCell(num_units=decoder_hidden_dim)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = work_nm_vocab_size)

    # decoder_true_input = tf.one_hot(work_nm, depth = work_nm_vocab_size, dtype=tf.float32)
    # sampling_prob = tf.train.exponential_decay()
    work_nm_len = tf.cast(work_nm_len, tf.int32)
    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=work_nm_embed,
                                                              sequence_length=work_nm_len,
                                                                 embedding=work_nm_W,
                                                              sampling_probability=0.0)

    my_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cell,
        helper=helper,
        initial_state=encoder_output_stateTuple)

    final_outputs, final_state, final_sequence_length = (
        tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                          maximum_iterations=maximum_iterations,
                                          impute_finished=False))

    model_output = tf.argmax(final_outputs.rnn_output, axis=2)

with tf.variable_scope('loss'):
    # weights = tf.cast(tf.sequence_mask(work_nm_len), tf.float32)
    # weights = tf.Print(weights, [weights])
    # work_nm_len = tf.Print(work_nm_len, [work_nm_len])
    weights = tf.sequence_mask(work_nm_len, dtype=tf.float32)
    # seq_loss = tf.contrib.seq2seq.sequence_loss(final_outputs.rnn_output, work_nm,
    #                                  weights, name='sequence_loss',
    #                                             average_across_timesteps=False,
    #                                             average_across_batch=False,
    #                                             )
    softmax = tf.nn.softmax(final_outputs.rnn_output)
    seq_loss = -tf.reduce_sum(tf.one_hot(work_nm, depth=work_nm_vocab_size) *
                  tf.log(tf.add(softmax, tf.constant(1e-12))))

    sum_work_nm_len = tf.cast(tf.add(tf.reduce_sum(work_nm_len), tf.constant(1,
                                                                             dtype=tf.int32)), tf.float32)
    seq_loss_mean = seq_loss / sum_work_nm_len


with tf.variable_scope('train'):
    # train_op = tf.train.AdamOptimizer().minimize(seq_loss_mean)

    optimizer = tf.train.AdamOptimizer()
    gradients, variables = zip(*optimizer.compute_gradients(seq_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 50.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

count = 0
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        while not coord.should_stop():

            # result = sess.run([upg_embed]) # (batch_size, seq_len, embd_dim)
            # print(result[0].shape)
            #
            # result = sess.run([eitem_no_embed]) # (batch_size, embd_dim)
            # print(result[0].shape)
            #
            # print('hey')
            # a = sess.run([work_nm, final_outputs.rnn_output])
            # print(a[0].shape)
            # print(a[1].shape)

            # print(sess.run(work_nm))
            #
            # result = sess.run([hw_key_en_nm_embed]) # (batch_size, seq_len, embd_dim)
            # print(result[0].shape)
            #
            # result = sess.run([hw_en_nm_embed]) # (batch_size, seq_len, embd_dim)
            # print(result[0].shape)
            #
            # result = sess.run([work_nm_embed]) # (batch_size, seq_len, embd_dim)
            # print(result[0].shape)
            #
            # result = sess.run([outputs, last_states])
            # print(result[0][:,-1,:].shape)
            # print('-------------------')
            # print(result[1].c.shape)
            #
            # break
            # result = sess.run([final_outputs, final_state, final_sequence_length])
            # print(result)

            _, output, loss_val, mean_lv, swnl, sample_model_output, sample_wnm, f_list = \
                sess.run([train_op, final_outputs.rnn_output,
                                                                         seq_loss,
                seq_loss_mean, sum_work_nm_len,
                                                                     model_output,
                                                                     work_nm,
                                                                     features_list])

            print(count, loss_val)

            output_pretty = np.array([[convert_dict['WORK_NM'][1][idx] for idx in x]
                                        for x in
                                      sample_model_output])

            target_pretty = [[convert_dict['WORK_NM'][1][idx] for idx in x] for x in
                             sample_wnm]

            for i, j in zip(output_pretty, target_pretty):
                print(i)
                print(j)
                break

            count += 1

    except tf.errors.OutOfRangeError:
        print('Done training --epoch limit reached')
        print(count)
    finally:
        coord.request_stop()
    coord.join(threads)



