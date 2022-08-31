import tensorflow as tf
def Pix2Vox_loss_fn(y_true, y_pred):
    # y_true = tf.squeeze(y_true, 4)
    # y_pred = tf.squeeze(y_pred, 4)
    # loss = -tf.reduce_sum((y_true * tf.math.log(y_pred)) + ((1 - y_true) * tf.math.log(1 - y_pred)), (1, 2, 3)) / (
    #       32 * 32 * 32)
    loss_fun = tf.keras.losses.BinaryCrossentropy()
    loss = loss_fun(y_true, y_pred)*10
    return loss

