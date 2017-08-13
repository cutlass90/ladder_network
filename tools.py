import numpy as np
from sklearn.model_selection import train_test_split

def print_stat(labels):
    [print('class_{0} = {1}'.format(i,v)) for i,v in enumerate(np.sum(labels,0))]

def split_data_set(class_distrib, samples, labels, verbose=False):
    """ Split provaided samples and labels according to class_distrib
        Args:
            samples: ndarray, 2D array of samples
            labels: ndarray, 2D array of one-hot labels
            class_distrib: dict, key - class 0, 1, ..., item - number of samples

        Return:
            split_samples: ndarray, requiered 2D array of samples
            split_labels: ndarray, required 2D array of labels
            rest_samples: ndarray, rested 2D array of samples
            rest_labels: ndarray, rested 2D array of labels

    """
    if verbose:
        print('Set summary:')
        print_stat(labels)
    for i,v in enumerate(np.sum(labels,0)):
        if class_distrib[i] > v:
            raise ValueError('There is not enough data for class {0},\
                request{1}, find {2}'.format(i, class_distrib[i], v))

    indxs = []
    for cl, n in class_distrib.items():
        indxs.append(np.random.permutation(np.nonzero(labels[:, cl] == 1)[0])[:n])
    indxs = np.concatenate(indxs)
    rest_indxs = np.setdiff1d(np.arange(labels.shape[0]), indxs)
    return samples[indxs], labels[indxs], samples[rest_indxs], labels[rest_indxs], 
