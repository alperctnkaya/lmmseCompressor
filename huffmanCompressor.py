import numpy as np
from dahuffman import HuffmanCodec


class huffmanCompressor:
    def __init__(self, model, quantized=False, k=0):

        self.modelSize = None
        self.compressedSize = None
        self.huffmanCoding(model, quantized, k)
        self.compressionRatio = self.modelSize / self.compressedSize

    def huffmanCoding(self, model, quantized, k):
        modelSize = 0
        compressedModelSize = 0

        for layer in model.layers:
            if not layer.weights:
                continue
            else:
                length = 1
                shape = layer.weights[0].shape
                for l in shape:
                    length = l * length

                w_vec = np.reshape(layer.weights[0].numpy(), (1, length))[0]

                codec = HuffmanCodec.from_data(w_vec[k:])

                encoded = codec.encode(w_vec[k:])
                compressedModelSize = compressedModelSize + len(encoded)

                if quantized:
                    modelSize = modelSize + ((len(w_vec) - k) * quantized) // 8
                else:
                    modelSize = modelSize + (len(w_vec) - k) * 4

        self.modelSize = modelSize
        self.compressedSize = compressedModelSize
