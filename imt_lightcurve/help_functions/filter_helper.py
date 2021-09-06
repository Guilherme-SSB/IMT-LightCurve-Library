import numpy as np

def expand_edges(array, numExpansion) -> np.ndarray:
    """
    Receive an array and expanded their edges.
    Given an array of length N, it returns 
    an array with length (N + 2*numExpansion).
    The procedure add a certain `numExpansion`
    points before the first point of the array, 
    and the same `numExpansion` after the last 
    point of the array. This values added are
    respectively equals to the first point and 
    the last point of the array.
    Parameters
    ----------
    array : numpy ndarray
        Array to be expanded.
    numExpansion : int
        Number of points to be added at the beginning 
        and at the end of the array.
    Returns
    -------
    `np.array`
    """
    aux_pre = np.zeros(numExpansion)
    aux_pos = np.zeros(numExpansion)
    i = 0

    for i in range(numExpansion):
        aux_pre[i] = array[0]
        aux_pos[i] = array[-1]

    return np.concatenate((aux_pre, array, aux_pos)).ravel()


def padding(array) -> np.ndarray:
    """
    Receive an array and apply the 
    zero padding algorithm.
    Given an array of length N, it returns 
    an array with length (2*N), filled 
    with zeros.
    Parameters
    ----------
    array : numpy ndarray
        Array to be zero padded.
    Returns
    -------
    `np.array`
    """
    return np.append(array, np.zeros(len(array)))


def centralize_fourier(array) -> np.ndarray:
    """
    Receive an array and multiply
    each value of it by the factor:
    (-1)^i, being i the array index.
    Parameters
    ----------
    array : numpy ndarray
        Array to be multiplied
    Returns
    -------
    `np.array`
    """
    multiplied = np.ones(len(array))
    i = 0

    for i in range(len(array)):
        multiplied[i] = array[i] * ((-1)**i)

    return multiplied


def fourier_transform(array) -> np.ndarray:
    """
    Receive an array and computes 
    the fourier transform of it.
    Parameters
    ----------
    array : numpy ndarray
        Array to be transformed
    Returns
    -------
    `np.array`
    """
    return np.fft.fft(array)


def inverse_fourier_transform(array) -> np.ndarray:
    """
    Receive an array and computes 
    the inverse fourier transform of it.
    Parameters
    ----------
    array : numpy ndarray
        Array to be [inversed] transformed
    Returns
    -------
    `np.array`
    """
    return np.real(np.fft.ifft(array))


def remove_padding(array) -> np.ndarray:
    """
    Receive an array and remove
    the padding transformation 
    previously applied.
    Parameters
    ----------
    array : numpy ndarray
        Array to lose their padded
    Returns
    -------
    `np.array`
    
    """
    return array[:int(len(array)/2)]


def remove_expand_edges(array, numExpansion) -> np.ndarray:
    """
    Receive an array and remove the 
    expantion edges.
    The description of what the expanded
    borded do is at `filter_helper.expand_edges`.
    This method undo this procedure.
    Parameters
    ----------
    array : numpy ndarray
        Array to lose the expansion.
    numExpansion : int
        Number of points previously choose.
    Returns
    -------
    `np.array`
    """
    aux = np.delete(array, np.s_[:numExpansion])

    return np.delete(aux, np.s_[-numExpansion:])