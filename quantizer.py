from utils import *
prune = True

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
                mean = np.mean(w_vec)

                if prune:
                    w_vec[np.where((w_vec < 0.2) * (w_vec > 0))[0]] = 0
                    w_vec[np.where((w_vec > -0.2) * (w_vec < 0))[0]] = 0

                partition, codebook = partition_codebook(w_vec, bits, uniform)

                i, quantized = quantize(w_vec, partition, codebook)
                w = np.reshape(quantized, shape)

                if True:
                    mean_diff = np.mean(w) - mean
                    w = w - mean_diff

                layer.set_weights([w, np.array(layer.weights[1].numpy())])

        return model