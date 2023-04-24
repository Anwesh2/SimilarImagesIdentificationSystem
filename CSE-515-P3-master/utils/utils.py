from enum import IntEnum
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

import functools
import numpy as np
import json
import pickle
import scipy as sp


class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (complex, np.complex)):
            return obj.real
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)


# Color Moment Models
class ColorMoment(IntEnum):
    MEAN = 1
    STD = 2
    SKEW = 3


# Feature Descriptor Models
class FDM(IntEnum):
    CM = 1
    ELBP = 2
    HOG = 3


def pca(data_matrix, k, reduce=False):
    data_matrix = np.nan_to_num(data_matrix, neginf=0, posinf=0)
    M = np.mean(data_matrix, axis=0)
    data_matrix_mean_centered = data_matrix - M
    data_matrix_mean_centered = np.nan_to_num(
        data_matrix_mean_centered, neginf=0, posinf=0)
    cov = np.cov(data_matrix_mean_centered, rowvar=False)
    cov = np.nan_to_num(cov, neginf=0, posinf=0)
    eigenValues, eigenVectors = np.linalg.eig(cov)
    eigenValues, eigenVectors = eigenValues.real, eigenVectors.real
    idx = eigenValues.argsort()[-k:][::-1]
    top_k_eigenValues = eigenValues[idx]
    top_k_eigenVectors = eigenVectors[:, idx]
    if reduce:
        ret = np.dot(top_k_eigenVectors.transpose(),
                     data_matrix_mean_centered.transpose())
        print(np.shape(ret))
        return ret
    return top_k_eigenVectors, top_k_eigenValues


def svd(data_matrix, k):
    data_matrix = np.nan_to_num(data_matrix, neginf=0, posinf=0)
    eigenValues, eigenVectors = np.linalg.eig(
        np.dot(np.transpose(data_matrix), data_matrix))
    eigenValues, eigenVectors = eigenValues.real, eigenVectors.real
    idx = eigenValues.argsort()[-k:][::-1]
    top_k_eigenValues = eigenValues[idx]
    top_k_eigenVectors = eigenVectors[:, idx]
    return top_k_eigenVectors, top_k_eigenValues


def lda(data_matrix, k):
    data_matrix = np.nan_to_num(data_matrix, neginf=0, posinf=0)
    lda = LatentDirichletAllocation(n_components=k)
    lda.fit(data_matrix)
    return np.transpose(lda.components_)


def kmeans(data_matrix, k):
    # TODO: Implement from scratch
    data_matrix = np.nan_to_num(data_matrix, neginf=0, posinf=0)
    kmeans = KMeans(n_clusters=k).fit(data_matrix)
    top_k = kmeans.cluster_centers_
    return np.transpose(top_k)


def get_image_from_dir(path):
    """Opens Image from given path

    Args:
        path (str): image path

    Returns:
        ndarray: Loaded Image
    """
    image = Image.open(path)
    return np.array(image)

# Function to calculate Chi-distance


def chisquare(A, B):
    """
    Calculates Chisquare distance
    """
    l = []
    for (a, b) in zip(A, B):
        if a + b != 0:
            l.append(((a - b) ** 2) / (a + b))
    chi = 0.5 * np.sum(l)
    return chi


def wasserstein(A, B):
    """
    Calculates wasserstein or earth movers distance
    """
    return sp.stats.wasserstein_distance(A, B)


def cosine(A, B):
    """
    Calculates cosine similarity distance
    """
    return sp.spatial.distance.cosine(A, B)


def euclidean(A, B):
    """
    Calculates euclidean similarity distance
    """
    A = np.array(A)
    B = np.array(B)
    return np.sqrt(np.sum((A - B)**2))
    # return np.linalg.norm(A-B)


def mean(pixels):
    """Calculates Mean

    Args:
        pixels (ndarray): pixels of image

    Returns:
        ndarray: Returns ndarray with mean
    """
    return np.mean(pixels)


def std(pixels):
    """Calculates Standard Deviation

    Args:
        pixels (ndarray): pixels of image

    Returns:
        ndarray: Returns ndarray with Standard Deviation
    """
    return np.std(pixels)


def skewness(pixels):
    """Calculates Skewness

    Args:
        pixels (ndarray): pixels of image

    Returns:
        ndarray: Returns ndarray with Skewness
    """
    return sp.stats.skew(pixels, axis=None)


def extended_local_binary_pattern(image):
    """
    This function returns feature discriptor vector of image by applying
    gray scale and rotation invariant uniform LBP and rotation invariant
    variance measures of the contrast of local image texture.

    Args:
        image (ndarray): Image on which ELBP needs to be applied

    Returns:
        ndarray: ELBP Feature descriptor vector of image
    """
    # settings for LBP
    # What is ideal radius to choose here?? refer these links
    # https://stackoverflow.com/questions/36607556/what-is-the-radius-parameter-for-in-opencvs-createlbphfacerecognizer
    # https://stackoverflow.com/questions/58049246/how-to-tune-local-binary-pattern
    # https://hindawi.com/journals/isrn/2013/429347/
    # radius = 1
    # n_points = 8 * radius

    # # print("UNIFORM LOCAL BINARY PATTERN...")
    # # improved rotation invariance with uniform patterns and
    # # finer quantization of the angular space which is gray scale and rotation invariant.
    # uniform_lbp1 = local_binary_pattern(image, n_points, radius, 'uniform')

    # # print("VARIANCE LOCAL BINARY PATTERN...")
    # # rotation invariant variance measures of the contrast of local
    # # image texture which is rotation but not gray scale invariant.
    # var_lbp1 = local_binary_pattern(image, n_points, radius, 'var')

    # # print("EXTENDED LOCAL BINARY PATTERN...")
    # # joint distribution with the local variance
    # elbp1 = np.concatenate((uniform_lbp1,
    #                         (var_lbp1*255)), axis=1)

    # radius = 2
    # n_points = 8 * radius
    # uniform_lbp2 = local_binary_pattern(image, n_points, radius, 'uniform')
    # var_lbp2 = local_binary_pattern(image, n_points, radius, 'var')
    # elbp2 = np.concatenate((uniform_lbp2,
    #                         (var_lbp2*255)), axis=1)

    # return elbp1 + elbp2
    radius = 4
    n_points = 8 * radius

    uniform_lbp = local_binary_pattern(image, n_points, radius, 'uniform')

    return uniform_lbp


def color_moment(image, cm_type):
    """
    This function returns color moment feature descriptor for given image.

    Args:
        image (ndarray): Image on which Color Moment needs to be applied
        cm_type (Enum): Color Moment calculation method i.e mean, std, skewness

    Returns:
        ndarray: Color Moment Feature descriptor vector of image
    """
    # print("{}: ".format(cm_type))
    moment = None
    if cm_type == ColorMoment.MEAN:
        moment = functools.partial(mean)
    elif cm_type == ColorMoment.STD:
        moment = functools.partial(std)
    else:
        moment = functools.partial(skewness)

    # Split image into 8*8 window and apply Color moment for each window
    unified_feature_descriptor = list()
    face_sections_rows = np.vsplit(image, 8)
    for i, face_section_row in enumerate(face_sections_rows):
        face_sections = np.hsplit(face_section_row, 8)
        unified_feature_descriptor.append(list())
        for face_section in face_sections:
            unified_feature_descriptor[i].append(moment(pixels=face_section))

    return np.array(unified_feature_descriptor)


def histogram_oriented_gradient(image):
    """Calculates feature descriptor of iamge using histogram of gradient

    Args:
        image (ndarray): Image pixels 64*64

    Returns:
        (ndarray, ndarray): Feature Descriptor and Actual image
    """
    fd, image = hog(image, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True, visualize=True)
    return fd, image


def save_to_file(data, filename, format="json"):
    if format == "json":
        with open(filename, "w+") as f:
            f.write(json.dumps(data, indent=2, cls=JsonCustomEncoder))
    elif format == "pickle":
        with open(filename, "wb+") as f:
            pickle.dump(data, f)
