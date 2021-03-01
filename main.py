from decompressor import decompressor
from compressor import compressor
from quantizer import quantizer
from huffmanCompressor import huffmanCompressor
from utils import *
from tensorflow.keras.datasets import cifar10

from tensorflow import keras

if __name__ == "__main__":
    modelPath = "lenet-300-100"
    model = keras.models.load_model(modelPath)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)

    num_classes = 10
    y_test = keras.utils.to_categorical(y_test, num_classes)


    print("Original Model:")
    model.evaluate(x_test, y_test)

    k = 16
    bits = 2
    uniform = False  #uniform or non uniform quantization of model weights
    block_size = 0

    c = compressor(model, k, bits, uniform, block_size)
    compressedModel = c.compressedModel
    parameters = c.parameters

    print("Compressed:")
    #h = huffmanCompressor(compressedModel, bits, k)
    #print("huffman Compression Ratio:", h.compressionRatio)

    d = decompressor(compressedModel, parameters, block_size)
    decompressedModel = d.decompressedModel

    decompressedModel.evaluate(x_test, y_test)

    model = keras.models.load_model(modelPath)
    q = quantizer(model, bits, uniform)
    quantizedModel = q.quantizedModel

    print("Quantized:")
    #h_q = huffmanCompressor(quantizedModel, bits, k)
    #print("huffman Compression Ratio:", h_q.compressionRatio)

    quantizedModel.evaluate(x_test, y_test)
