import numpy as np
np.set_printoptions(suppress=True)
import struct
file1 = 'train-images.idx3-ubyte'
file2 = 'train-labels.idx1-ubyte'

def loadImageSet(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)

    labelNum = head[1]
    offset = struct.calcsize('>II')

    numString = '>' + str(labelNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)

    binfile.close()
    labels = np.reshape(labels, [labelNum])

    return labels, head


def show_trans(x):
    return 1 if x > 0 else 0
show_trans = np.vectorize(show_trans)


def load_data(file_img, file_lab, batch_size):
    imgs, data_head = loadImageSet(file_img)
    labels_temp, labels_head = loadLabelSet(file_lab)
    labels = np.zeros((batch_size, 10))
    for i in range(batch_size):
        labels[i, labels_temp[i]] = 1
    return normalization(imgs[0:batch_size, :]), labels[0:batch_size]


def load_label(file_lab, batch_size):
    labels_temp, labels_head = loadLabelSet(file_lab)
    return labels_temp[0:batch_size]


def normalization(x):
    return x / 255

# X, Y = load_data(file1, file2, 64)
# print(X.shape, Y.shape)
# print(Y[0:4, :])


# if __name__ == "__main__":
#     file1 = 'train-images.idx3-ubyte'
#     file2 = 'train-labels.idx1-ubyte'
#
#     imgs, data_head = loadImageSet(file1)
#     print(imgs.shape)
#     # print('data_head:', data_head)
#     # print(type(imgs))
#     # print('imgs_array:', imgs)
#     img_raw = np.reshape(imgs[0, :], [28, 28])
#     imgs_show = show_trans(img_raw)
#     # np.savetxt("training_data_64.txt", imgs[0:64, :], fmt="%d", delimiter=" ")
#     print('--------------------------------')
#
#     labels, labels_head = loadLabelSet(file2)
#     # np.savetxt("training_label_64.txt", labels[0:64], fmt="%d", delimiter=" ")
#     # print('labels_head:', labels_head)
#     # print(type(labels))
#     print(labels[0])
#     # print(labels.shape)

