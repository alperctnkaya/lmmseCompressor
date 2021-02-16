from utils import *


class decompressor:
    def __init__(self, model, a):
        self.decompressedModel = self.decompress(model,a)

    def decompress(self, model, a):
        a = iter(a)

        for layer in model.layers:
            if not layer.weights:
                continue
            else:
                length = 1
                shape = layer.weights[0].shape
                for l in shape:
                    length = l * length

                w_vec = np.reshape(layer.weights[0].numpy(), (1, length))[0]
                decoded = self.decode(w_vec, next(a))
                w = np.reshape(decoded, shape)

                layer.set_weights([w, np.array(layer.weights[1].numpy())])

        return model

    @classmethod
    def decode(cls, vec, a):
        k = len(a)
        decoded = np.zeros(len(vec))
        decoded[: k] = vec[: k]

        for i in range(k, len(vec)):
            decoded[i] = vec[i - k:i] @ a[::-1] + vec[i]
            vec[i] = decoded[i]

        return decoded
