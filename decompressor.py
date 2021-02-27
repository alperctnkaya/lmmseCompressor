from utils import *


class decompressor:
    def __init__(self, model, parameters):
        self.decompressedModel = self.decompress(model,parameters)

    def decompress(self, model, paramters):
        one_to_many = True

        a = paramters["a"]
        means = paramters["means"]
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
                    w_vec = np.reshape(layer.weights[0].numpy(), (1, length))[0]
                else:
                    if len(shape) == 4:
                        w_vec = np.reshape(np.transpose(layer.weights[0].numpy(), (0, 1, 3, 2)), (1, length))[0]
                    else:
                        w_vec = np.reshape(np.transpose(layer.weights[0].numpy()), (1, length))[0]

                decoded = self.decode(w_vec, next(a))

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
