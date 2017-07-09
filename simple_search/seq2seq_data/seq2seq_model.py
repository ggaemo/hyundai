import tensorflow as tf


class EmbeddingModel():

    def __init__(self, decoder_hidden_dim,
                 batch_size,
                 embedding_dim,
                 maximum_iterations,
                 convert_dict,
                 inputs, is_train):
        self.decoder_hidden_dim = decoder_hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.maximum_iterations = maximum_iterations
        self.eitem_rnn_hidden_dim = 32

        for key in convert_dict:
            convert_dict[key][1][0] = '_'
        self.convert_dict = convert_dict

        self.work_nm_vocab_size = len(convert_dict['WORK_NM'][0].keys())
        self.train_op, self.seq_loss_mean, self.model_output, self.target = \
            self.build_model(inputs, is_train)

    def build_model(self, inputs, is_train):

        global_step = tf.Variable(0, name='global_step', trainable=False)

        def get_embedding_variable(var_name, embedding_size, inputs):
            vocab_size = len(self.convert_dict[var_name][0])
            with tf.variable_scope('embedding_layer'):
                variable_embeddings = tf.get_variable \
                    (name='variable_embeddings_{}'.format(var_name),
                     shape=[vocab_size, embedding_size],
                     initializer=tf.random_uniform_initializer(-1, 1))

                embed_variable = tf.nn.embedding_lookup(variable_embeddings, inputs,
                                                        name='variable_lookup_{}'.format(
                                                            var_name))
            return variable_embeddings, embed_variable

        upg_no, eitem_no, hw_key_en_nm, hw_key_en_nm_len, hw_en_nm, hw_en_nm_len, \
        work_nm, work_nm_len = inputs

        self.upg_no = upg_no
        self.eitem_no = eitem_no
        self.hw_key_en_nm = hw_key_en_nm
        self.hw_en_nm = hw_en_nm

        _, upg_embed = get_embedding_variable('UPG_NO', self.embedding_dim, upg_no)

        _, eitem_no_embed = get_embedding_variable('E_ITEM_NO', self.embedding_dim, eitem_no)

        _, hw_key_en_nm_embed = get_embedding_variable('HW_KEY_EN_NM_LIST', self.embedding_dim,
                                                       hw_key_en_nm)
        _, hw_en_nm_embed = get_embedding_variable('HW_EN_NM_LIST', self.embedding_dim,
                                                   hw_en_nm)
        work_nm_W, work_nm_embed = get_embedding_variable('WORK_NM', self.embedding_dim, work_nm)

        features_list = list()


        # with tf.variable_scope('upg_no'):
        #     upg_no_feature = tf.reduce_sum(upg_embed, axis=1)
        #     features_list.append(upg_no_feature)

        # with tf.variable_scope('upg_no'):
        #     rnn_cell = tf.contrib.rnn.LSTMCell(num_units=self.upg_rnn_hidden_dim)
        #
        #     rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
        #         rnn_cell,
        #         attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(6, upg_embed),
        #         attention_layer_size=32,
        #         alignment_history=False)
        #
        #     # outputs, last_states = tf.nn.dynamic_rnn(cell=rnn_cell,
        #     #                                          inputs=upg_embed,
        #     #                                          dtype=tf.float32
        #     #                                          )
        #
        #     (outputs, output_state_fw, output_state_bw) = \
        #         tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        #             cells_fw=[rnn_cell],
        #             cells_bw=[rnn_cell],
        #             inputs=upg_embed,
        #             dtype=tf.float32)
        #
        #     upg_feature = tf.concat([outputs[:,0,:], outputs[:,-1,:]], axis=1)
        #     #last output
        #     # upg_feature = tf.concat([output_state_fw[-1].c, output_state_bw[-1].c],
        #     #                         axis=1)  #
        #     # last output
        #     #  or tf.concat([outputs[:,-1,:], last_states.c], axis=1)
        #     features_list.append(upg_feature)

        with tf.variable_scope('eitem_no'):
            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=self.eitem_rnn_hidden_dim)
            # outputs, last_states = tf.nn.dynamic_rnn(cell=rnn_cell,
            #                                          inputs=upg_embed,
            #                                          dtype=tf.float32
            #                                          )
            # eitem_no_feature = outputs[:, -1, :]
        #
            (outputs, output_state_fw, output_state_bw) = \
                tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=[rnn_cell],
                    cells_bw=[rnn_cell],
                    inputs=upg_embed,
                    dtype=tf.float32)

            eitem_no_feature = tf.concat([outputs[:, 0, :], outputs[:, -1, :]], axis=1)
            features_list.append(eitem_no_feature)

        # with tf.variable_scope('eitem_no'):
        #     eitem_no_feature = eitem_no_embed
        #     features_list.append(eitem_no_feature)


        with tf.variable_scope('hw_key_en_nm'):
            weights = tf.sequence_mask(hw_key_en_nm_len, dtype=tf.float32)
            weights = tf.expand_dims(weights, 2)
            hw_key_en_nm_embed = tf.multiply(hw_key_en_nm_embed, weights)
            hw_key_en_nm_feature = tf.reduce_sum(hw_key_en_nm_embed, axis=1)
            features_list.append(hw_key_en_nm_feature)
        #
        with tf.variable_scope('hw_en_en_nm'):
            weights = tf.sequence_mask(hw_en_nm_len, dtype=tf.float32)
            weights = tf.expand_dims(weights, 2)
            hw_en_nm_embed = tf.multiply(hw_en_nm_embed, weights)
            hw_en_nm_feature = tf.reduce_sum(hw_en_nm_embed, axis=1)
            features_list.append(hw_en_nm_feature)

        encoder_output = tf.concat(features_list, axis=1)


        output_c = tf.contrib.layers.fully_connected(encoder_output,
                                                     num_outputs=self.decoder_hidden_dim,
                                                     activation_fn = tf.tanh
                                                     )
        output_gate = tf.contrib.layers.fully_connected(encoder_output,
                                                     num_outputs=self.decoder_hidden_dim,
                                                     activation_fn = tf.sigmoid
                                                     )
        output_m = tf.tanh(output_c) * output_gate

        # output_c = tf.zeros_like(output_m)

        # output_m = tf.zeros_like(output_c)

        encoder_output_stateTuple = tf.contrib.rnn.LSTMStateTuple(output_c, output_m)

        with tf.variable_scope('decoder'):
            cell = tf.contrib.rnn.LSTMCell(num_units=self.decoder_hidden_dim,
                                           use_peepholes=False)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,
                                                          output_size=self.work_nm_vocab_size)

            # decoder_true_input = tf.one_hot(work_nm, depth = work_nm_vocab_size, dtype=tf.float32)
            if is_train:
                self.sampling_prob = tf.train.inverse_time_decay(learning_rate=1.0,
                                                                 global_step=global_step,
                                                                 decay_steps=1000,
                                                                 decay_rate=0.9)
                self.sampling_prob = tf.constant(1.0) - self.sampling_prob
                # self.sampling_prob = tf.constant(1.0)
            else:
                self.sampling_prob = tf.constant(1.0)
            work_nm_len = tf.cast(work_nm_len, tf.int32)
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=work_nm_embed,
                                                                         sequence_length=work_nm_len,
                                                                         embedding=work_nm_W,
                                                                         sampling_probability=self.sampling_prob) #TODO

            my_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell,
                helper=helper,
                initial_state=encoder_output_stateTuple)

            # my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            #     cell=cell,
            #     embedding = work_nm_W,
            #     helper=helper,
            #     initial_state=encoder_output_stateTuple)

            final_outputs, final_state, final_sequence_length = (
                tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                  maximum_iterations=self.maximum_iterations,
                                                  parallel_iterations = 32))

            model_output = tf.argmax(final_outputs.rnn_output, axis=2)

        with tf.variable_scope('loss'):
            # weights = tf.sequence_mask(work_nm_len, dtype=tf.float32)
            # seq_loss = tf.contrib.seq2seq.sequence_loss(final_outputs.rnn_output, work_nm,
            #                                  weights, name='sequence_loss',
            #                                             average_across_timesteps=False,
            #                                             average_across_batch=False,
            #                                             )
            # TODO NAN이 발생해....


            softmax = tf.nn.softmax(final_outputs.rnn_output)

            # seq_loss = -tf.reduce_sum(tf.one_hot(work_nm,
            #                                        depth=self.work_nm_vocab_size) *
            #                           tf.log(tf.add(softmax, tf.constant(1e-12))),
            #                           axis=[1,2])
            #
            # seq_loss_mean = tf.reduce_mean(seq_loss)

            # sum_work_nm_len = tf.cast(tf.add(tf.reduce_sum(work_nm_len),
            #                                  tf.constant(1,dtype=tf.int32)),tf.float32)
            #
            # seq_loss_mean = seq_loss / sum_work_nm_len

            seq_loss_mean = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(work_nm, depth=self.work_nm_vocab_size),
                    logits=final_outputs.rnn_output))

            # seq_loss_mean = -tf.reduce_mean(tf.one_hot(work_nm,
            #                                        depth=self.work_nm_vocab_size) *
            #                           tf.log(tf.add(softmax, tf.constant(1e-12))))

        with tf.variable_scope('metrics'):
            self.acc = tf.reduce_mean(tf.cast(tf.equal(model_output,
                                                 work_nm),
                                  tf.float32))

        if is_train:
            with tf.variable_scope('train'):
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(seq_loss_mean, global_step=global_step)
        else:
            train_op = _

            # optimizer = tf.train.AdamOptimizer()
            # gradients, variables = zip(*optimizer.compute_gradients(seq_loss))
            # print('clip gradients 50.0')
            # gradients, _ = tf.clip_by_global_norm(gradients, 50.0)
            # train_op = optimizer.apply_gradients(zip(gradients, variables))

        return train_op, seq_loss_mean, model_output, work_nm