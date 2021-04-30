import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from data import num_classes, gen_synthetic_data
from model import ResNet50Trunc
from loss import embedding_loss


image, label = gen_synthetic_data()
image_shape = image.shape

model = ResNet50Trunc(image_shape, 8, False)

image_tf = tf.expand_dims(tf.convert_to_tensor(image / 255.), axis=0)
label_tf = tf.expand_dims(tf.convert_to_tensor(label), axis=0)

##############################################
# kmeans on embedded features before training.
##############################################
feature_np = model(image_tf).numpy()
feature_flat = feature_np.reshape((image_shape[0] * image_shape[1], 8))
label_flat = label.flatten()
feature_mask = feature_flat[label_flat > 0]

kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(feature_mask)
labels_pred = np.zeros_like(label_flat)
labels_pred[label_flat > 0] = kmeans.labels_ + 1
labels_pred = labels_pred.reshape((image_shape[0], image_shape[1]))

plt.figure()
plt.imshow(labels_pred)


##############################################
# kmeans on embedded features after training.
##############################################
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        feature = model(image_tf)
        loss = embedding_loss(feature, label_tf, num_classes)

    grads = tape.gradient(loss, model.model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.model.trainable_weights))

    if i % 20 == 0:
        print(loss)

feature_np = model(image_tf).numpy()
feature_flat = feature_np.reshape((image_shape[0] * image_shape[1], 8))
label_flat = label.flatten()
feature_mask = feature_flat[label_flat > 0]

kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(feature_mask)
labels_pred = np.zeros_like(label_flat)
labels_pred[label_flat > 0] = kmeans.labels_ + 1
labels_pred = labels_pred.reshape((image_shape[0], image_shape[1]))

plt.figure()
plt.imshow(labels_pred)
plt.show()
