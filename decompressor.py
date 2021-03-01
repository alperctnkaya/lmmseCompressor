from utils import *


class decompressor:
    def __init__(self, model, parameters, block_size=0):
        self.decompressedModel = self.decompress(model,parameters, block_size)

    def decompress(self, model, parameters, block_size):
        one_to_many = False

        a = parameters["a"]
        means = parameters["means"]
        a = iter(a)
        means = iter(means)

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

                if (block_size == 0) or (block_size > len(vec)):
                    _block_size = len(vec)
                else:
                    _block_size = block_size

                decoded = []

                for i in range(int(np.ceil(len(vec) / _block_size))):
                    w_vec = vec[i * _block_size:(i + 1) * _block_size]
                    decoded.append(self.decode(w_vec, next(a)))

                decoded = [j for sub in decoded for j in sub]

                if one_to_many:
                    w = np.reshape(decoded, shape)

                else:
                    if len(shape) == 4:

                        shape = list(shape)
                        shape[2:] = shape[2:][::-1]
                        w = np.transpose(np.reshape(decoded, shape), (0, 1, 3, 2))

                    else:
                        w = np.transpose(np.reshape(decoded, shape[::-1]), (1, 0))

                mean_diff = np.mean(w) - next(means)
                w = w - mean_diff


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
