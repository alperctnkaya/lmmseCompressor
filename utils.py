import numpy as np
from sklearn.cluster import KMeans

def quantize(signal, partitions, codebook):
    indices = []
    quanta = []
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        indices.append(index)
        quanta.append(codebook[index])
    return indices, quanta


def partition_codebook(vec, bits, uniform = True):
    if uniform:
        minVec = np.min(vec)
        maxVec = np.max(vec)

        stepSize = (maxVec - minVec) / 2**bits

        codebook = np.arange(minVec,maxVec, stepSize)
        partition = codebook[1:]

    else:
        Kmeans = KMeans(n_clusters = 2**bits)
        Kmeans.fit(np.reshape(vec,(-1,1)))

        codebook = np.reshape(np.sort(Kmeans.cluster_centers_, axis=0),(2**bits,))
        partition = codebook[1:]

    return partition, codebook


def mse_bw_models(model1, model2):
    for w1, w2 in zip(model1.weights, model2.weights):
        print(np.mean((w1-w2)**2))