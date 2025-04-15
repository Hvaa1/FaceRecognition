
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
    results = []
    verification_images = os.listdir(os.path.join("application_data","verification_images"))
    input_img_path = os.path.join("application_data","input_images","input_image.jpg")
    input_img = preprocess(input_img_path)
    for image in verification_images:
        validation_img= preprocess(os.path.join('application_data', 'verification_images', image))
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)),verbose=0)
        results.append(result)
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold
    return results, verified
