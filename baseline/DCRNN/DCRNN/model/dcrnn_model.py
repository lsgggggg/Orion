from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from lib.metrics import masked_mae_loss
from model.dcrnn_cell import DCGRUCell

class DCRNNModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))

        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')

        # Labels: (batch_size, timesteps, num_sensor, output_dim)
        # 修改：使用 output_dim 而不是 input_dim
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, output_dim), name='labels')

        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes, output_dim))

        cell = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_with_projection = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type)
        encoding_cells = [cell] * num_rnn_layers
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        encoding_cells = tf.nn.rnn_cell.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.nn.rnn_cell.MultiRNNCell(decoding_cells, state_is_tuple=True)

        global_step = tf.train.get_or_create_global_step()

        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            # Encoder
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            _, enc_state = tf.nn.static_rnn(encoding_cells, inputs, dtype=tf.float32)

            # Decoder
            labels = tf.unstack(
                tf.reshape(self._labels, (batch_size, horizon, num_nodes * output_dim)), axis=1)
            labels.insert(0, GO_SYMBOL)

            # 自定义解码器逻辑
            outputs = []
            state = enc_state
            for t in range(horizon):
                if is_training:
                    # Scheduled sampling
                    if use_curriculum_learning:
                        c = tf.random.uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        input_t = tf.cond(tf.less(c, threshold), lambda: labels[t], lambda: outputs[-1] if t > 0 else GO_SYMBOL)
                    else:
                        input_t = labels[t]
                else:
                    input_t = outputs[-1] if t > 0 else GO_SYMBOL

                output_t, state = decoding_cells(input_t, state)
                outputs.append(output_t)

            # Project the output to output_dim.
            outputs = tf.stack(outputs, axis=1)  # [batch_size, horizon, num_nodes * output_dim]
            self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')

        self._merged = tf.summary.merge_all()

        if is_training:
            # Loss
            scaler = self._scaler
            loss_fn = masked_mae_loss(scaler, null_val=0.0)
            self._loss = loss_fn(self._outputs, self._labels)
            self._mae = self._loss

            # Optimization
            optimizer = tf.train.AdamOptimizer(learning_rate=model_kwargs.get('base_lr', 0.01))
            tvars = tf.trainable_variables()
            grads = tf.gradients(self._loss, tvars)
            clipped_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
            self._train_op = optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=global_step)

            # Log merge for tensors (e.g., gradients)
            self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs