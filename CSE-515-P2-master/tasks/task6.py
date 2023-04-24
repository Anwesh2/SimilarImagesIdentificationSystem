
from numpy.lib import utils
from utils.utils import pca, svd, lda, kmeans, save_to_file, euclidean

import os
import numpy as np
import pickle
import json

from utils.utils import get_image_from_dir
from .phase1 import get_feature_descriptor, task3, show_olivetti_faces
from .task5 import get_nd_of_image


def dim_reduce(dr, k, img):
    if dr == "pca":
        current_img_latent_semantics, _ = pca(img, k)
    elif dr == "svd":
        current_img_latent_semantics, _ = svd(img, k)
    elif dr == "lda":
        current_img_latent_semantics = lda(img, k)
    else:
        current_img_latent_semantics = kmeans(img, k)
    return current_img_latent_semantics


def transform_to_latent_space(latent_semantics, img_fd):
    return np.matmul(latent_semantics, img_fd)


def run(args):
    data = {}
    with open(args.latent_semantic_file, "rb") as f:
        data = pickle.load(f)
    args.fd_model = data["fd-model"]
    args.cm_type = data["cm-type"]
    k = data["k"]

    # Find model and load corresponding feature descriptor
    model = args.fd_model
    task2_out = os.path.join(args.image_dir,
                             "fd", model)
    if args.fd_model == "cm":
        task2_out = os.path.join(task2_out, args.cm_type)
        model = "{}-{}".format(model, args.cm_type)
    task2_out = os.path.join(task2_out, "fd.pkl")
    with open(task2_out, "rb") as f:
        images = pickle.load(f)

    # Specially handled for phase-2
    latent_semantics = data["top-k-vectors"]
    query_image_fd = None
    if args.query_image_file not in images:
        test_image = get_image_from_dir(os.path.join(args.query_image_file))
        _, fd = get_feature_descriptor(
            args, test_image)
        if args.fd_model == "hog":
            query_image_fd = fd[1]
        else:
            query_image_fd = fd
    transformed_query_image = None
    if data["task"] == "task-3" or data["task"] == "task-4":
        transformed_query_image = transform_to_latent_space(get_nd_of_image(
            query_image_fd, data, images), latent_semantics)
    else:
        transformed_query_image = transform_to_latent_space(
            np.array(query_image_fd).ravel(), latent_semantics)
        transformed_query_image = np.nan_to_num(
            transformed_query_image, posinf=0, neginf=0)
    for img_name in images:
        if data["task"] == "task-3" or data["task"] == "task-4":
            images[img_name] = transform_to_latent_space(
                data["nd-data"][img_name], latent_semantics)
        else:
            images[img_name] = transform_to_latent_space(
                np.array(images[img_name]).ravel(), latent_semantics)
            images[img_name] = np.nan_to_num(
                images[img_name], posinf=0, neginf=0)

    type_score = {}
    for img_name in images:
        xyz = img_name.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        dist = euclidean(transformed_query_image, images[img_name])
        if x not in type_score:
            type_score[x] = [dist]
        type_score[x].append(dist)

    min = float('inf')
    type_label = "NA"
    for type in type_score:
        mean = np.mean(type_score[type])
        if mean < min:
            min = mean
            type_label = type
        type_score[type] = mean
    print("TYPE LABEL: {} with average score/distance of {}".format(type_label, min))
    print("All type average distance scores from query image:")
    print(dict(sorted(type_score.items(), key=lambda item: item[1])))
