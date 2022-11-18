import os
import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2


def load_data(path, split=0.1):
    # 80-10-10
    # Sorting it so that training images can pair with their respective masks.
    images = sorted(glob(os.path.join(path,"Original\*")))
    masks = sorted(glob(os.path.join(path,"Ground Truth\*")))

    total_size = len(images)
    valid_size = int(split*total_size)
    test_size = int(split*total_size)

    train_x, valid_x = train_test_split(images,test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks,test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x,test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y,test_size=test_size, random_state=42)

    return (train_x,train_y),(valid_x,valid_y),(test_x,test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x,(256,256))
    # Normailizing it to reduce the computation time.
    x = x/255.0
    # (256x256x3)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x,(256,256))
    # Normailizing it to reduce the computation time.
    x = x/255.0
    # (256,256) -> (256,256,1)
    x = np.expand_dims(x,axis=-1)
    return x

def tf_parse(x,y):
    def _parse(x,y):
        x = read_image(x)
        y = read_mask(y)
        return x,y
    x, y = tf.numpy_function(_parse, [x,y], [tf.float64, tf.float64])
    x.set_shape([256,256,3])
    y.set_shape([256,256,1])
    print(x.shape,y.shape)
    return x,y


def tf_dataset(x,y,batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

if __name__=='__main__':
    path = "CVC-ClinicDB"
    ((train_x,train_y),(valid_x,valid_y),(test_x,test_y)) = load_data(path)
    print(len(train_x),len(valid_x),len(test_x))

    ds = tf_dataset(test_x,test_y)

    for x,y in ds:
        print(x.shape,y.shape)
        break

