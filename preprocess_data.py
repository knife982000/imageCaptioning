import json
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import math


BATCH = 16


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


if __name__ == '__main__':
    dataset = 'train'
    annotation_file = 'annotations/captions_{}2014.json'.format(dataset)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    path = '{}2014'.format(dataset) + os.sep
    image_paths = []
    images_ids = list(set([x['image_id'] for x in annotations['annotations']]))
    images_ids.sort()
    for img in images_ids:
        image_paths.append(path + 'COCO_{}2014_'.format(dataset) + '%012d.jpg' % (img))

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    output_path = '{}_processed2014'.format(dataset)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path in tqdm(image_dataset, total=math.ceil(len(image_paths) / BATCH)):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            path_of_feature = output_path + os.sep + os.path.basename(path_of_feature)
            np.save(path_of_feature, bf.numpy())
