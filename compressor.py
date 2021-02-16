from utils import *
from decompressor import decompressor


class compressor:

    def __init__(self, model=None, k=None, bits=None, uniform = True):

        if model is not None:
            self.a=[]
            self.compressedModel = self.compress(model, k, bits, uniform)

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

    def compress(self, model, k, bits, uniform):

        for layer in model.layers:
            if not layer.weights:
                continue
            else:
                length = 1
                shape = layer.weights[0].shape
                for l in shape:
                    length = l * length

                w_vec = np.reshape(layer.weights[0].numpy(), (1, length))[0]

                Rx, MRx = self.calcAutoCorrelationMatrix(w_vec, k)
                a = np.linalg.solve(MRx, Rx[1:])
                self.a.append(a)
                diff, prediction = self.predict(w_vec, a, k)
                partition, codebook = partition_codebook(diff, bits, uniform)
                encoded = self.encode(w_vec, a, k, codebook, partition)

                w = np.reshape(encoded, shape)

                layer.set_weights([w, np.array(layer.weights[1].numpy())])

        return model
