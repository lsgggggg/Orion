# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : base_model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion
# @Modified : Mar. 31, 2025 by Grok 3 (xAI)

import tensorflow as tf
import os
from models.layers import *

def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    """
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of ST-Conv blocks.
    :param keep_prob: placeholder.
    """
    x = inputs

    # Ensure the input channel matches the first block's input channel
    C_0 = inputs.get_shape().as_list()[-1]  # Get the input channel number
    assert C_0 == blocks[0][0], f"Input channel {C_0} does not match the first block's input channel {blocks[0][0]}"

    # ST-Conv blocks
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')

    # Output layer
    if len(blocks) > 1:
        x = output_layer(x, 1, blocks[-1][-1], 1, 'sigmoid')  # Correct parameter order
    else:
        x = output_layer(x, 1, blocks[0][-1], 1, 'sigmoid')  # Correct parameter order

    train_loss = tf.reduce_mean(tf.square(x - inputs[:, n_his:n_his + 1, :, 0:1]))
    tf.add_to_collection(name='y_pred', value=x)

    return train_loss, x

def model_save(sess, global_step, model_name, save_path):
    """
    Save the model checkpoint.
    :param sess: TensorFlow session.
    :param global_step: Global step number.
    :param model_name: Name of the model.
    :param save_path: Path to save the model.
    """
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(save_path, model_name)
    saver.save(sess, checkpoint_path, global_step=global_step)
    print(f"Model saved to {checkpoint_path} at step {global_step}")