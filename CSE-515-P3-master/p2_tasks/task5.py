
from numpy.lib import utils
from utils.utils import pca, svd, lda, kmeans, save_to_file

import os
import numpy as np
import pickle
import json

from utils.utils import get_image_from_dir, euclidean
from .phase1 import get_feature_descriptor, task3, show_olivetti_faces


def run(args):
    data = {}
    with open(args.latent_semantic_file, "rb") as f:
        data = pickle.load(f)
    args.k = args.n
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
    query_image_fd = None
    # if args.query_image_file not in images:
    test_image = get_image_from_dir(os.path.join(args.query_image_file))
    _, fd = get_feature_descriptor(
        args, test_image)
    if args.fd_model == "hog":
        query_image_fd = fd[1]
    else:
        query_image_fd = fd
    # latent_semantics = data["top-k-vectors"]
    rank = {}
    latent_semantics = data["top-k-vectors"]
    if data["task"] == "task-3" or data["task"] == "task-4":
        query_image_in_latent_space = transform_to_latent_space(get_nd_of_image(
            query_image_fd, data, images), latent_semantics)
    for img_name in images:
        xyz = img_name.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        if data["task"] == "task-3" or data["task"] == "task-4":
            current_image_in_latent_space = transform_to_latent_space(
                data["nd-data"][img_name], latent_semantics)
        else:
            current_image_fd = np.nan_to_num(
                images[img_name].ravel(), posinf=0, neginf=0)
            current_image_in_latent_space = np.matmul(
                current_image_fd, latent_semantics)
            current_image_in_latent_space = np.nan_to_num(
                current_image_in_latent_space)
            query_image_in_latent_space = np.matmul(
                np.asarray(query_image_fd).ravel(), latent_semantics)
        rank[img_name] = euclidean(
            current_image_in_latent_space, query_image_in_latent_space)
    rank = dict(sorted(rank.items(), key=lambda item: item[1]))
    count = 0
    image_list = [args.query_image_file]
    neigh_dist = [0.0]
    original_faces = [get_image_from_dir(os.path.join(args.query_image_file))]
    print("Input image: {}".format(args.query_image_file))
    print("Image        <=::::::::::=>      Distance")
    for img in rank:
        print("{}<=:::=>{}".format(img, rank[img]))
        image_list.append(img)
        neigh_dist.append(rank[img])
        original_faces.append(get_image_from_dir(
            os.path.join(args.image_dir, img)))
        count = count + 1
        if count >= args.n:
            break

    show_olivetti_faces(original_faces, image_list, neigh_dist,
                        "top n images")()


def get_nd_of_image(fd, data, images):
    fd_nd = []
    dist_map = {}
    for img_name in images:
        xyz = img_name.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        if data["task"] == "task-4":
            # if x != "original":
            #     continue
            x = y
        if x in dist_map:
            dist_map[x].append(euclidean(np.array(fd), images[img_name]))
        else:
            dist_map[x] = [euclidean(np.array(fd), images[img_name])]
    for type in data["similarity-list"]:
        fd_nd.append(np.sum(dist_map[type]) / len(dist_map[type]))
    return fd_nd


def transform_to_latent_space(latent_semantics, img_fd):
    return np.matmul(latent_semantics, img_fd)
