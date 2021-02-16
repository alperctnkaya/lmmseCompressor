from utils import *

class quantizer:
    def __init__(self, model, bits, uniform):
        self.quantizedModel = self.quantizeModel(model, bits, uniform)

    def quantizeModel(self, model, bits, uniform):
        for layer in model.layers:
            if not layer.weights:
                continue
            else:
                length = 1
                shape = layer.weights[0].shape
                for l in shape:
                    length = l * length

                w_vec = np.reshape(layer.weights[0].numpy(), (1, length))[0]
                partition, codebook = partition_codebook(w_vec, bits, uniform)

                i, quantized = quantize(w_vec, partition, codebook)
                w = np.reshape(quantized, shape)

                layer.set_weights([w, np.array(layer.weights[1].numpy())])

        return model