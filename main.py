from decompressor import decompressor
from compressor import compressor
from quantizer import quantizer
from huffmanCompressor import huffmanCompressor
from utils import *
from tensorflow.keras.datasets import cifar10

from tensorflow import keras

if __name__ == "__main__":
    modelPath = "mnist_conv_small"
    model = keras.models.load_model(modelPath)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)

    num_classes = 10
    y_test = keras.utils.to_categorical(y_test, num_classes)


    print("Original Model:")
    model.evaluate(x_test, y_test)

    k = 32
    bits = 4
    uniform = True  #uniform or non uniform quantization of model weights

    c = compressor(model, k, bits, uniform)
    compressedModel = c.compressedModel
    parameters = c.parameters

    h = huffmanCompressor(compressedModel, bits, k)
    print("Compressed:")
    print("huffman Compression Ratio:", h.compressionRatio)

    d = decompressor(compressedModel, parameters)
    decompressedModel = d.decompressedModel

    decompressedModel.evaluate(x_test, y_test)

    model = keras.models.load_model(modelPath)
    q = quantizer(model, bits, uniform)
    quantizedModel = q.quantizedModel

    h_q = huffmanCompressor(quantizedModel, bits, k)
    print("Quantized:")
    print("huffman Compression Ratio:", h_q.compressionRatio)

    quantizedModel.evaluate(x_test, y_test)
