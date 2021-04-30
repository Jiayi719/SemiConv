import tensorflow as tf


# Loss function from eq.(5).
def embedding_loss(feature, label_gt, num_classes):
    # Assume id=0 is background.
    loss_total = 0.
    for i in range(1, num_classes + 1):
        mask = tf.equal(label_gt, i)
        num_pixels_i = tf.reduce_sum(tf.cast(mask, tf.float32))

        feature_mask = tf.boolean_mask(feature, mask)
        centroid = tf.reduce_sum(feature_mask, axis=[0]) / num_pixels_i
        centroid = tf.reshape(centroid, (1, 1, 1, tf.shape(centroid)[0]))

        loss_i = tf.boolean_mask(feature - centroid, mask)
        loss_i = tf.norm(loss_i, axis=1)
        loss_i = tf.reduce_sum(loss_i) / num_pixels_i
        loss_total += loss_i

    loss_total /= num_classes
    return loss_total
