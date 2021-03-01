from utils import *
from decompressor import decompressor

prune = False

class compressor:

    def __init__(self, model=None, k=None, bits=None, uniform=True, block_size=0):

        if model is not None:
            self.parameters = {}
            self.parameters["a"] = []
            self.parameters["means"] = []
            self.compressedModel = self.compress(model, k, bits, uniform, block_size)

    @classmethod
    def calcAutoCorrelationMatrix(cls, vec, k):
        Rx = np.zeros(k + 1)
        lenVec = len(vec)
        for j in range(0, k + 1, 1):
            i = np.arange(0, len(vec) - k)
            Rx[j] = vec[i + j] @ np.transpose(vec[i]) / lenVec

        MRx = np.zeros((k, k))
        Rx_1 = Rx[0:len(Rx) - 1]

        for i in range(0, k, 1):
            MRx[i][:] = np.concatenate((Rx_1[i::-1], Rx_1[1:len(Rx) - i - 1]), 0)

        return Rx, MRx

    @classmethod
    def predict(cls, vec, a, k):
        prediction = np.zeros(len(vec) - k)

        for i in range(k, len(vec), 1):
            prediction[i - k] = vec[i - k:i] @ a[::-1]

        diff = vec[k:] - prediction

        return diff, prediction

    def encode(self, vec, a, k, codebook, partition):
        encoded = np.zeros(len(vec))
        encoded[:k] = vec[: k]

        qErr = np.zeros(len(vec) - k)

        for i in range(k, len(vec)):
            prediction = vec[i - k:i] @ a[::-1]
            err = vec[i] - prediction
            [index, quantizedErr] = quantize(np.array([err]), partition, codebook)
            vec[i] = prediction + quantizedErr[0]

            qErr[i - k] = quantizedErr[0]

        encoded = np.concatenate((vec[:k], qErr), axis=0)

        return encoded

    def compress(self, model, k, bits, uniform, block_size):
        one_to_many = False  # defines the way of serializing weight matrix

        for layer in model.layers:
            if not layer.weights:
                continue
            else:
                length = 1
                shape = layer.weights[0].shape
                for l in shape:
                    length = l * length

                if one_to_many:
                    vec = np.reshape(layer.weights[0].numpy(), (1, length))[0]
                else:
                    if len(shape) == 4:
                        vec = np.reshape(np.transpose(layer.weights[0].numpy(), (0, 1, 3, 2)), (1, length))[0]
                    else:
                        vec = np.reshape(np.transpose(layer.weights[0].numpy()), (1, length))[0]

                self.parameters["means"].append(np.mean(vec))


                if (block_size == 0) or (block_size > len(vec)):
                    _block_size = len(vec)
                else:
                    _block_size = block_size

                if prune:
                    vec[np.where((vec < 0.1) * (vec > 0))[0]] = 0
                    vec[np.where((vec > -0.1) * (vec < 0))[0]] = 0
                    print(len(np.where(vec == 0)[0]))

                encoded = []

                for i in range(int(np.ceil(len(vec) / _block_size))):

                    w_vec = vec[i*_block_size:(i+1) * _block_size]
                    Rx, MRx = self.calcAutoCorrelationMatrix(w_vec, k)

                    a = np.linalg.solve(MRx, Rx[1:])
                    self.parameters["a"].append(a)
                    diff, prediction = self.predict(w_vec, a, k)
                    partition, codebook = partition_codebook(diff, bits, uniform)

                    encoded.append(self.encode(w_vec, a, k, codebook, partition))

                encoded = [j for sub in encoded for j in sub]



                if one_to_many:
                    w = np.reshape(encoded, shape)
                else:
                    if len(shape) == 4:
                        shape = list(shape)
                        shape[2:] = shape[2:][::-1]
                        w = np.transpose(np.reshape(encoded, shape), (0, 1, 3, 2))
                    else:
                        w = np.transpose(np.reshape(encoded, shape[::-1]), (1, 0))

                layer.set_weights([w, np.array(layer.weights[1].numpy())])

        return model
