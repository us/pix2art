import cv2
import os
import numpy as np
from os import listdir
import random
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Pix2Art AML Final Project')
parser.add_argument('-p','--dataset-path',dest="dataset_path", type=str, help='Dataset path for cleaning')
parser.add_argument('-o','--output-path',dest="output_path", type=str, help='Dataset output path')
parser.add_argument('-r', '--run',dest="run", help='Run directly from webcam stream', action='store_true')

args = parser.parse_args()

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


# holistically-nested edge detection
# takes cv2 image
# generates edge detected form of that image
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


# crop image to 256x256 pixels
def cropImage(img):
    img_map = img[0:256, 0:256]
    return img_map


# return files inside a given folder
def folderList(path: str):
    import glob
    return glob.glob(path)


# resize images to be fitted horizontally 256 pixels
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


# resize, crop and apply hed method in this order
# then concatenate hed image and cropped size side by side
def convertImages(path, outputPath):
    for index, imageFile in enumerate(folderList(path + "/*.jpg")):
        img = cv2.imread(imageFile)
        resizedImage = image_resize(img)
        croppedImage = cropImage(resizedImage)
        hedImage = hedConvert(croppedImage)
        hedImage = np.stack((hedImage, ) * 3, axis=-1)
        concatedImage = np.concatenate((hedImage, croppedImage), axis=1)
        cv2.imwrite(outputPath + "/" + str(index + 1) + ".jpg", concatedImage)
        print(imageFile.split("/")[-1])


# shuffle files inside the a given directory
def shuffle(path):
    print(len(listdir(path)))
    for k in range(10):
        for i in listdir(path):
            r = str(random.randint(1, 1360))
            if r == i.split('.')[0]:
                r = str(random.randint(1, 1360))
            if os.path.exists(path + i):
                os.rename(path + r + '.jpg', path + 'test.jpg')
                os.rename(path + i, path + r + '.jpg')
                os.rename(path + 'test.jpg', path + i)


# Load a (frozen) Tensorflow model into memory.
def load_graph(frozen_graph_filename):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

# apply data cleaning and preprocessing methods to webcam feed
def runWebcamStream():
    # use webcam capture device
    cap = cv2.VideoCapture(0)
    graph = load_graph("frozen_model.pb")
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)
    while True:
        ret, frame = cap.read()
        resizedImage = image_resize(frame)
        croppedImage = cropImage(resizedImage)
        hedImage = hedConvert(croppedImage)
        hedImage = np.stack((hedImage, ) * 3, axis=-1)
        hedImage = np.concatenate((hedImage, hedImage), axis=1)
        generated_image = sess.run(output_tensor,
                                   feed_dict={image_tensor: hedImage})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image),
                                 cv2.COLOR_RGB2BGR)
        concatedImage = np.concatenate((image_bgr, croppedImage), axis=1)

        # Display the resulting frame
        cv2.imshow('frame', concatedImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    if args.run:
        print("Running from webcam stream")
        runWebcamStream()
    elif args.dataset_path!=None or args.output_path!=None:
    	print(f"Cleaning dataset from {args.dataset_path} to {args.output_path}")
    	convertImages(args.dataset_path, args.output_path)

