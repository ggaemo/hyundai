import pickle
import re
import os
import pandas as pd
import tensorflow as tf


data_dir = '/home/jinwon/PycharmProjects/hyundai/simple_search/seq2seq_data/'

def run_word_to_idx(word_frequency_threshold, work_nm_len_threshold):

    def preprocess(df):

        eng_nans = df[df['HW_KEY_EN_NM_LIST'].apply(lambda x: True if 'nan' in x else False)]
        eng_ko = []
        for i in range(len(eng_nans)):
            tmp_ser = pd.Series(eng_nans['HW_KEY_EN_NM_LIST'].iloc[i])
            tmp_ser_ko = pd.Series(eng_nans['HW_KEY_KO_NM_LIST'].iloc[i])
            tmp_ser[tmp_ser_ko[tmp_ser == 'nan'].index] = tmp_ser_ko[tmp_ser == 'nan']
            eng_ko.append(list(tmp_ser))
        eng_nans.loc[:, 'HW_KEY_EN_NM_LIST'] = pd.Series(eng_ko, index=eng_nans.index)
        return df.drop(eng_nans.index).append(eng_nans).sort_index()

    df_train = pd.read_pickle(data_dir + 'df_train.df')
    df_train = df_train.reset_index(drop=True)
    df_train.fillna('nan', inplace=True)
    df_train.index = ['trn_{}'.format(x) for x in df_train.index]


    df_test = pd.read_pickle(data_dir +'df_test.df')
    df_test = df_test.reset_index(drop=True)
    df_test.fillna('nan', inplace=True)
    df_test.index = ['test_{}'.format(x) for x in df_test.index]

    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    df_all = pd.concat([df_train, df_test])

    df_all['UPG_NO'] = df_all['UPG_NO'].apply(func = lambda x: x.replace('-', ''))


    ###filter data
    cond = df_all['UPG_NO'].apply(lambda x: bool(re.search('\d{3}', x)))

    df_all = df_all.loc[cond]

    df_all['UPG_NO'] = df_all['UPG_NO'].apply(lambda x: re.search('(\d{3})', x).group(1))
    # df_all = df_all.loc[df_all['UPG_NO'] != '999']

    train = pd.DataFrame()

    train['UPG_NO'] = df_all['UPG_NO'].apply(lambda x: [x])

    # train['UPG_NO'] = df_all['UPG_NO'].apply(list)
    # train['UPG_NO'] = train['UPG_NO'].apply(lambda x: ['{}_{}'.format(elem, idx) for
    #                                                    idx,elem in enumerate(x)])

    train['E_ITEM_NO'] = df_all['E_ITEM'].apply(lambda x: x[:10])

    train['E_ITEM_NO'] = train['E_ITEM_NO'].apply(list)

    train.loc[train['E_ITEM_NO'].apply(len) == 0, 'E_ITEM_NO'] = ['empty']
    # train['E_ITEM_NO'] = train['E_ITEM_NO'].apply(lambda x: [x])
    # train['E_ITEM_NM_KO'] = df_all['E_ITEM_NM_KO'].apply(lambda x: [x])
    # train['E_ITEM_NM_EN'] = df_all['E_ITEM_NM_EN'].apply(lambda x: [x])

    df_all['HW_EN_NM_LIST'] = df_all['HW_EN_NM_LIST'].apply(lambda x: ['empty'] if
    len(x) == 0 else x)
    df_all['HW_KEY_EN_NM_LIST'] = df_all['HW_KEY_EN_NM_LIST'].apply(lambda x: [
        'empty'] if len(x) == 0 else x)



    train['HW_EN_NM_LIST'] = df_all['HW_EN_NM_LIST']

    train['HW_KEY_EN_NM_LIST'] = df_all['HW_KEY_EN_NM_LIST']

    train['WORK_NM'] = df_all['WORK_NM']

    work_nm_len = train['WORK_NM'].apply(len)

    train = train.loc[work_nm_len <= work_nm_len_threshold]
    def make_dictionary(df, frequency_threshold):
        df_copy = df.copy(deep=True)
        convert_col = dict()

        for col in df_copy:
            unk_val = 'UNK_{}'.format(col)
            all_values = list()
            df_copy[col].apply(lambda x: all_values.extend(x))
            value_count = pd.Series(all_values).value_counts()
            sufficient_val = value_count.loc[value_count >
                                             frequency_threshold].index.values
            # print('unk if empty')
            df_copy[col] = df_copy[col].apply(lambda x: [elem if bool(elem in sufficient_val) else
                                              unk_val for elem in x])
            # df_copy[col] = df_copy[col].apply(
            #     lambda x: [elem  for elem in x if bool(elem in sufficient_val)])
            # df_copy[col] = df_copy[col].apply(lambda x: [unk_val] if len(x) == 0 else x)

        for col in df_copy:
            unique_values = set()
            # df.loc[df[col] == 'nan', col] = 'UNK_{}'.format(col)
            df_copy[col].apply(lambda x: unique_values.update(x))
            word_idx = dict()
            for idx, word in enumerate(unique_values, start=1):
                word_idx[word] = idx
            df_copy[col] = df_copy[col].apply(lambda row: [word_idx[elem] for elem in row])
            idx_word = {word : idx for idx, word in word_idx.items()}
            convert_col[col] = (word_idx, idx_word)
        return df_copy, convert_col


    df, convert_dict = make_dictionary(train, word_frequency_threshold)

    with open(data_dir + 'convert_dict.pkl', 'wb') as f:
        pickle.dump(convert_dict, f)

    df_train = df.loc[[x.startswith('trn_') for x in df.index.values]]
    df_test = df.loc[[x.startswith('test_') for x in df.index.values]]
    # for i in range(10):
    #     df_train.sample(n=len(df_train)).to_pickle(data_dir +
    #                                                'idx_df_train_random_sample_{}.df'.format(i))
    i=0
    df_train.sample(n=len(df_train)).to_pickle(data_dir +
                                                   'idx_df_train_random_sample_{}.df'.format(i))

    df_test.to_pickle(data_dir + 'idx_df_test.df')


if not os.path.exists(data_dir+'idx_df_train_random_sample_0.df'):
    run_word_to_idx(10, 12)


def run_tfrecords_generation():


    def make_example(UPG_NO, E_ITEM_NO, HW_KEY_EN_NM_LIST, HW_EN_NM_LIST, WORK_NM):
        ex = tf.train.SequenceExample()

        f_upg_no = ex.feature_lists.feature_list['UPG_NO']
        f_eitem_no = ex.feature_lists.feature_list['E_ITEM_NO']
        f_hw_key_en_nm = ex.feature_lists.feature_list['HW_KEY_EN_NM_LIST']
        f_en_nm = ex.feature_lists.feature_list['HW_EN_NM_LIST']
        f_work_nm = ex.feature_lists.feature_list['WORK_NM']

        c_key_en_len = ex.context.feature['HW_KEY_EN_NM_LIST_LEN']
        c_en_len = ex.context.feature['HW_EN_NM_LIST_LEN']
        c_work_nm_len = ex.context.feature['WORK_NM_LEN']

        def add_seq(f, value_list):
            for val in value_list:
                f.feature.add().int64_list.value.append(val)
            return f

        def add_context(f, value):
            f.int64_list.value.append(value)

        f_upg_no = add_seq(f_upg_no, UPG_NO)
        f_eitem_no = add_seq(f_eitem_no, E_ITEM_NO)
        f_hw_key_en_nm = add_seq(f_hw_key_en_nm, HW_KEY_EN_NM_LIST)
        f_en_nm = add_seq(f_en_nm, HW_EN_NM_LIST)
        f_work_nm = add_seq(f_work_nm, WORK_NM)

        c_key_en_len = add_context(c_key_en_len, len(HW_KEY_EN_NM_LIST))
        c_en_len = add_context(c_en_len, len(HW_EN_NM_LIST))
        c_work_nm_len = add_context(c_work_nm_len, len(WORK_NM))

        return ex


    for filename in os.listdir(data_dir):
        match = re.match('(idx_df_train_random_sample_\d).df', filename)
        if match:
            filename = os.path.join(data_dir+match.group(1)+'.tfrecords')
            writer = tf.python_io.TFRecordWriter(filename)
            data = pd.read_pickle(data_dir+match.group())
            for idx, row in data.iterrows():
                ex = make_example(**row)
                writer.write(ex.SerializeToString())
            writer.close()

    for filename in os.listdir(data_dir):
        match = re.match('(idx_df_test).df', filename)
        if match:
            filename = os.path.join(data_dir+match.group(1)+'.tfrecords')
            writer = tf.python_io.TFRecordWriter(filename)
            data = pd.read_pickle(data_dir+match.group())
            for idx, row in data.iterrows():
                ex = make_example(**row)
                writer.write(ex.SerializeToString())
            writer.close()

if not os.path.exists(data_dir+'idx_df_train_random_sample_0.tfrecords'):
    run_tfrecords_generation()


def read_and_decode(filename_queue):
    print('Reading and Decoding')
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
    'HW_KEY_EN_NM_LIST_LEN' : tf.FixedLenFeature([], dtype=tf.int64),
    'HW_EN_NM_LIST_LEN' : tf.FixedLenFeature([], dtype=tf.int64),
    'WORK_NM_LEN' : tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "UPG_NO": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "E_ITEM_NO": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "HW_KEY_EN_NM_LIST": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "HW_EN_NM_LIST": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "WORK_NM": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }



    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        context_features=context_features,
        serialized=serialized_example,
        sequence_features=sequence_features
        )
    return sequence_parsed, context_parsed


def inputs(data_name_list, batch_size, num_epochs, num_threads=1):
    filename = [os.path.join(data_dir, data_name) for data_name
                in data_name_list]
    filename_queue = tf.train.string_input_producer(filename, num_epochs)
    reader_sequence_output, reader_context_output = read_and_decode(filename_queue)
    # print('transposed')
    upg_no = reader_sequence_output['UPG_NO']
    eitem_no = reader_sequence_output['E_ITEM_NO']
    hw_key_en_nm = reader_sequence_output['HW_KEY_EN_NM_LIST']
    hw_en_nm = reader_sequence_output['HW_EN_NM_LIST']
    work_nm = reader_sequence_output['WORK_NM']

    hw_key_en_nm_len = reader_context_output['HW_KEY_EN_NM_LIST_LEN']
    hw_en_nm_len = reader_context_output['HW_EN_NM_LIST_LEN']
    work_nm_len = reader_context_output['WORK_NM_LEN']



    batch = tf.train.batch([upg_no, eitem_no, hw_key_en_nm, hw_key_en_nm_len ,hw_en_nm,
                            hw_en_nm_len,
                            work_nm,
                            work_nm_len],
                             batch_size, dynamic_pad=True, allow_smaller_final_batch=False,
                           capacity = batch_size * 2, num_threads=num_threads)

    return batch


def run_test():
    batch_size = 128
    num_epochs = 10
    with tf.Graph().as_default():
        data_name_list = []
        for filename in os.listdir(data_dir):
            match = re.match('idx_df_train_random_sample_\d.tfrecords', filename)
            if match:
                data_name_list.append(match.group())
        a = inputs(data_name_list, batch_size, num_epochs)
        count = 0
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            try:
                while not coord.should_stop():
                    upg_no, eitem_no, hw_key_en_nm, hw_en_nm, work_nm, work_nm_len = sess.run(a)

                    print('-------------------')
                    count += 1

            except tf.errors.OutOfRangeError:
                print('Done training --epoch limit reached')
                print(count)
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    # run_test()
    print('hello')
