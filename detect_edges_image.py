import cv2
import os
import numpy as np
from os import listdir
import random


# import tensorflow as tf


class CropLayer(object):
    def __init__(self, params, blobs):

        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]


def hedConvert(image):

    protoPath = os.path.sep.join(["hed_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(
        ["hed_model", "hed_pretrained_bsds.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    cv2.dnn_registerLayer("Crop", CropLayer)

    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image,
                                 scalefactor=1.0,
                                 size=(W, H),
                                 mean=(104.00698793, 116.66876762,
                                       122.67891434),
                                 swapRB=False,
                                 crop=False)

    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    cv2.dnn_unregisterLayer("Crop")
    return hed


def cropImage(img):
    img_map = img[0:256, 0:256]
    return img_map


def folderList(path: str):
    import glob
    return glob.glob(path)


def image_resize(image, width=256, height=256, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if h < w:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def convertImages(path):
    for index, imageFile in enumerate(folderList(path + "/*.jpg")):
        img = cv2.imread(imageFile)
        resizedImage = image_resize(img)
        croppedImage = cropImage(resizedImage)
        hedImage = hedConvert(croppedImage)
        hedImage = np.stack((hedImage, ) * 3, axis=-1)
        concatedImage = np.concatenate((hedImage, croppedImage), axis=1)
        cv2.imwrite("dataset/" + str(index + 1) + ".jpg", concatedImage)
        print(imageFile.split("/")[-1])


def shuffle(path):
    print(len(listdir(path)))

    for k in range(10):
        for i in listdir(path):
            r = str(random.randint(1, 1360))
            if r == i.split('.')[0]:
                r = str(random.randint(1, 1360))
            if os.path.exists(path+i):
                os.rename(path + r + '.jpg', path + 'test.jpg')
                os.rename(path + i, path + r + '.jpg')
                os.rename(path + 'test.jpg', path+i)

    # for i in range(1, 8027)[-400:]:
        # os.rename(path+str(i)+".jpg", "dataset/test/"+str(i)+".jpg")

    # path = 'dataset/test/'

    # for i, f in enumerate(listdir(path)):
    #     os.rename(path + f, path + str(i+1) + ".jpg")

#cap = cv2.VideoCapture(-1)
#cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
#cap.set()

def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

# cap = cv2.VideoCapture(0)
if __name__ == "__main__":
    # graph = load_graph("frozen_model.pb")
    # image_tensor = graph.get_tensor_by_name('image_tensor:0')
    # output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    # sess = tf.Session(graph=graph)
    # while True:
    #     ret, frame = cap.read()
    #     resizedImage = image_resize(frame)
    #     croppedImage = cropImage(resizedImage)
    #     hedImage = hedConvert(croppedImage)
    #     hedImage = np.stack((hedImage, ) * 3, axis=-1)
    #     hedImage = np.concatenate((hedImage, hedImage), axis=1)
    #     generated_image = sess.run(output_tensor, feed_dict={image_tensor: hedImage})
    #     image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
    #     concatedImage = np.concatenate((image_bgr, croppedImage), axis=1)


    #     # Display the resulting frame
    #     cv2.imshow('frame', concatedImage)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

    #convertImages("jpg")
    shuffle("dataset/")