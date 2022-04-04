import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import struct
from array import array
import os

def load_binary(img_path, lbl_path):
    with open(lbl_path, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number should be 2049, got {magic}"
        labels = array("B", f.read())

    with open(img_path, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number should be 2051, got {magic}"
        image_data = array("B", f.read())

    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        images[i][:] = image_data[i * rows * cols : (i + 1) * rows * cols]

    return images, labels

def load_mnist(pca=False, n_components=8):
    train_img_path = r"data/train-images.idx3-ubyte"
    train_lbl_path = r"data/train-labels.idx1-ubyte"

    train_img, train_lbl = load_binary(train_img_path, train_lbl_path)

    test_img_path = r"data/t10k-images.idx3-ubyte"
    test_lbl_path = r"data/t10k-labels.idx1-ubyte"

    test_img, test_lbl = load_binary(test_img_path, test_lbl_path)

    X_ = np.vstack((train_img, test_img))
    y_ = np.hstack((train_lbl, test_lbl))

    if pca:
        pca_ = PCA(n_components=n_components)
        X_ = pca_.fit_transform(X_)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tf = scaler.fit_transform(X_)

    X = (X_tf > 0.5).astype(bool)
    y = (y_ == 5) | (y_ == 6) | (y_ == 7) | (y_ == 8) | (y_ == 9)

    X_train = X[:60_000]
    X_test = X[60_000:]
    y_train = y[:60_000]
    y_test = y[60_000:]

    return X_train, X_test, y_train, y_test


# Main part of code below comes from https://tinyurl.com/4t7ec7pm
def import_binary_mnist(image_path, label_path):
    f = open(image_path, "rb")
    magic, size, width, height = struct.unpack(">IIII", f.read(16))
    data_image = array("B", f.read())
    f.close()

    images = []
    for i in range(size):
        images.append([0] * height * width)

    for i in range(size):
        images[i][:] = data_image[i * height * width : (i + 1) * height * width]

    images = np.array(images, dtype=bool)

    with open(label_path, "rb") as f:
        # Without this line below, the label array will have 8 entries too much
        magic, size = struct.unpack(">II", f.read(8))
        labels = array("B", f.read())

    labels = np.array(labels, dtype=bool)

    return images, labels


def export_binary_mnist(X: np.ndarray, y: np.ndarray, image_name: str, label_name: str):
    assert len(X.shape) == 2, f"X must have shape (N, width*height)"
    assert X.shape[0] == y.shape[0], f"X and y must have the same size."
    size = X.shape[0]
    assert np.sqrt(X.shape[1]).is_integer(), f"Only square images are supported."
    width, height = [int(np.sqrt(X.shape[1]))] * 2

    data_image = array("B")
    for i in X.astype(int).flatten():
        data_image.append(i)

    data_label = array("B")
    for i in y.astype(int).flatten():
        data_label.append(i)

    hexval = "{0:#0{1}x}".format(size,6) # number of files in HEX

    # header for label array
    header = array('B')
    header.extend([0,0,8,1,0,0])
    header.append(int('0x'+hexval[2:][:2],16))
    header.append(int('0x'+hexval[2:][2:],16))

    data_label = header + data_label

    # additional header for images array
    # Maximum image size is 256x256
    header.extend([0,0,0,width,0,0,0,height])
    header[3] = 3 # Changing MSB for image data (0x00000803)
    data_image = header + data_image

    image_path = image_name + '-idx3-ubyte'
    output_file = open(image_path, 'wb')
    data_image.tofile(output_file)
    output_file.close()

    label_path = label_name + '-idx1-ubyte'
    output_file = open(label_path, 'wb')
    data_label.tofile(output_file)
    output_file.close()

    # gzip resulting files, omitting that for now
    # os.system('gzip '+image_name+'-idx3-ubyte')
    # os.system('gzip '+label_name+'-idx1-ubyte')

    # Checking if we exported correctly
    X_, y_ = import_binary_mnist(image_path, label_path)
    assert np.allclose(X, X_), f"Exported images do not match original"
    assert np.allclose(y, y_), f"Exported labels do not match original"
