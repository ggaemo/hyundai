import pickle
import pandas as pd
import numpy as np
from sequence_tfrecords import inputs
import tensorflow as tf
import os
import re
import seq2seq_model


data_config = '10_12'
data_dir = '/home/jinwon/PycharmProjects/hyundai/simple_search/seq2seq_data/'

save_dir = '/home/jinwon/PycharmProjects/hyundai/simple_search/seq2seq_data/{}/'.format(
    data_config)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir+'saved')

pd.set_option('expand_frame_repr', False)


def pretty_print(model_output, target, meta_data=None):
    output_pretty = np.array([[convert_dict['WORK_NM'][1][idx] for idx in x]
                              for x in
                              model_output])

    target_pretty = np.array([[convert_dict['WORK_NM'][1][idx] for idx in x]
                              for x in
                              target])

    df = pd.DataFrame(columns=['Model Output', 'Target'], dtype='object')

    for i in range(len(output_pretty)):
        df.loc[i, 'Model Output'] = output_pretty[i]

    for i in range(len(target_pretty)):
        df.loc[i, 'Target'] = target_pretty[i]

    if meta_data:
        for key, val in meta_data.items():
            # if key == 'E_ITEM_NO':
            #     column_val = np.array(
            #         [convert_dict[key][1][x] for x in val])
            # else: column_val = np.array([[convert_dict[key][1][idx] for idx in x] for x in val])

            column_val = np.array([[convert_dict[key][1][idx] for idx in x] for x in val])
            df[key] = None
            df = df.astype('object')
            for i in range(len(target_pretty)):
                df.loc[i, key] =  column_val[i]
        count = 0
        while os.path.exists(save_dir+'model_output_{}.df'.format(count)):
            count += 1
        df.to_pickle(save_dir+'model_output_{}.df'.format(count))

    print(df.head(20))

with open(data_dir + 'convert_dict.pkl', 'rb') as f:
    convert_dict = pickle.load(f)

decoder_hidden_dim = 32
batch_size = 512
test_batch_size = 2151
num_epochs = 1000
embedding_dim = 64
maximum_iterations = 12
# upg_rnn_hidden_dim = 32

trn_data_name_list = []
for filename in os.listdir(data_dir):
    match = re.match('idx_df_train_random_sample_\d.tfrecords', filename)
    if match:
        trn_data_name_list.append(match.group())

test_data_name_list = []
for filename in os.listdir(data_dir):
    match = re.match('idx_df_test.tfrecords', filename)
    if match:
        test_data_name_list.append(match.group())


with tf.name_scope('Train'):
    train_inputs = inputs(trn_data_name_list, batch_size, num_epochs, num_threads=1)
    with tf.variable_scope('Model', reuse=None):
        trn_model = seq2seq_model.EmbeddingModel(decoder_hidden_dim, batch_size, embedding_dim,
                                   maximum_iterations, convert_dict, train_inputs,
                                                 True)

with tf.name_scope('Test'):
    test_inputs = inputs(test_data_name_list, test_batch_size, num_epochs, num_threads=1)
    with tf.variable_scope('Model', reuse=True):
        test_model = seq2seq_model.EmbeddingModel(decoder_hidden_dim, batch_size, embedding_dim,
                                   maximum_iterations, convert_dict, test_inputs,
                                                 False)


saver = tf.train.Saver()

count = 0


test_loss_list = list()
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

            # _, output, loss_val, mean_lv, swnl, sample_model_output, sample_wnm, f_list = \
            #     sess.run([train_op, final_outputs.rnn_output,
            #                                                              seq_loss,
            #     seq_loss_mean, sum_work_nm_len,
            #                                                          model_output,
            #                                                          work_nm,
            #                                                          features_list])
            #

            if count % 50 == 0 and count > 2500:
                trn_loss, trn_model_output, trn_target, trn_samp_prob, trn_acc = sess.run([
                    trn_model.seq_loss_mean,
                    trn_model.model_output,
                                                   trn_model.target,
                                                         trn_model.sampling_prob,
                    trn_model.acc])


                upg_no, eitem_no, hw_key_en_nm, hw_en_nm, test_loss, test_model_output, \
                test_target, test_sampl_prob, test_acc = \
                    sess.run(
                    [test_model.upg_no, test_model.eitem_no, test_model.hw_key_en_nm ,test_model.hw_en_nm,
                     test_model.seq_loss_mean,
                     test_model.model_output, test_model.target,
                     test_model.sampling_prob, test_model.acc])

                print('\n\n')
                print(count, 'Train Loss', trn_loss, 'Test Loss', test_loss)
                print('Train----------------{}'.format(trn_samp_prob))
                pretty_print(trn_model_output, trn_target)
                print('Test---------------------------'.format(test_sampl_prob))
                pretty_print(test_model_output, test_target)

                print(trn_acc, test_acc)
                saver.save(sess, save_dir + 'saved/model_{}_{}.ckpt'.format(trn_loss,
                                                                       test_loss))

                test_loss_list.append(test_loss)


                if count % 1000 == 0:
                    meta_dict = dict()
                    meta_dict['UPG_NO'] = upg_no
                    meta_dict['E_ITEM_NO'] = eitem_no
                    meta_dict['HW_KEY_EN_NM_LIST'] = hw_key_en_nm
                    meta_dict['HW_EN_NM_LIST'] = hw_en_nm
                    pretty_print(test_model_output, test_target, meta_dict)




            else:
                sess.run([trn_model.train_op])
                # print(count, 'Train Loss', trn_loss, end='\r')

            count += 1

    except KeyboardInterrupt:
        trn_loss, trn_model_output, trn_target, trn_samp_prob, trn_acc = sess.run([
            trn_model.seq_loss_mean,
            trn_model.model_output,
            trn_model.target,
            trn_model.sampling_prob,
            trn_model.acc])

        upg_no, eitem_no, hw_key_en_nm, hw_en_nm, test_loss, test_model_output, \
        test_target, test_sampl_prob, test_acc = \
            sess.run(
                [test_model.upg_no, test_model.eitem_no, test_model.hw_key_en_nm,
                 test_model.hw_en_nm,
                 test_model.seq_loss_mean,
                 test_model.model_output, test_model.target,
                 test_model.sampling_prob, test_model.acc])

        meta_dict = dict()
        meta_dict['UPG_NO'] = upg_no
        meta_dict['E_ITEM_NO'] = eitem_no
        meta_dict['HW_KEY_EN_NM_LIST'] = hw_key_en_nm
        meta_dict['HW_EN_NM_LIST'] = hw_en_nm
        pretty_print(test_model_output, test_target, meta_dict)


    except tf.errors.OutOfRangeError:
        print('Done training --epoch limit reached')
        print(count)

    finally:
        coord.request_stop()
    coord.join(threads)



