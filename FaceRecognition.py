
import tensorflow as tf
import numpy as  np
import math
import os


def load_model(model):
    siamese_model = tf.keras.models.load_model(
        model,
        custom_objects={'L1Dist': L1Dist, 'binary_cross_loss': binary_cross_loss, 'SqueezeLayer': SqueezeLayer}
    )
    return siamese_model


class L1Dist(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__()
        def call(self, input_embedding, validation_embedding):
           return tf.math.abs([a - b for a, b in zip(input_embedding, validation_embedding)])


binary_cross_loss = tf.losses.BinaryCrossentropy()


class SqueezeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.squeeze(inputs, axis=0)


def preprocess(img_path):
    byte_img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img,(100,100))
    img = img[:,:,:3]
    img = img/255
    return img

def verify(model, detection_threshold, verification_threshold):
    verification_dir = os.path.join("application_data", "verification_images")
    verification_images = os.listdir(verification_dir)

    input_img_path = os.path.join("application_data", "input_images", "input_images.jpg")
    input_img = preprocess(input_img_path)

    # Tiền xử lý tất cả ảnh so sánh trước
    validation_imgs = [preprocess(os.path.join(verification_dir, img)) for img in verification_images]

    # Tạo batch input: (num_samples, 100, 100, 3)
    input_imgs_batch = np.array([input_img for _ in validation_imgs])
    validation_imgs_batch = np.array(validation_imgs)

    # Gom cặp input-validation cho model Siamese
    model_input = [input_imgs_batch, validation_imgs_batch]

    # Predict batch một lần duy nhất
    with tf.device('/GPU:0'):
        results = model.predict(model_input, verbose=0)

    # Tính toán kết quả
    detection = np.sum(results > detection_threshold)
    verification = detection / len(verification_images)
    verified = verification > verification_threshold

    return results, verified
