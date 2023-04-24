
from numpy.core.numeric import Inf, Infinity
from utils.utils import pca, svd, lda, kmeans, save_to_file

import os
import numpy as np
import pickle
from .task1 import get_subject_weight_pair


def run(args):
    k = args.k
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

    # Group fd per type
    feed_fd = {}
    feed_fd_2 = []
    for idx, i in images.items():
        xyz = idx.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        if y == str(args.y):
            feed_fd_2.append(list(np.array(i).ravel()))
            if x in feed_fd:
                feed_fd[x].append(list(np.array(i).ravel()))
            else:
                feed_fd[x] = list([np.array(i).ravel()])

    type_weight_pair = {}
    file = os.path.join(args.image_dir, "phase-2/task-2-output")
    os.makedirs(file, exist_ok=True)
    filename = None

    # Apply Dimension reductionality
    top_k_eigen_values = []
    if args.dim_reduce == "pca":
        # https://medium.com/@devalshah1619/face-recognition-using-eigenfaces-technique-f221d505d4f7
        # paper: http://www.mit.edu/~9.54/fall14/Classes/class10/Turk%20Pentland%20Eigenfaces.pdf
        top_k_vectors, top_k_eigen_values = pca(feed_fd_2, k)
        filename = os.path.join(
            file, "PCA-{}-subject-{}-top-{}-type-weights.txt".format(model, args.y, args.k))
    elif args.dim_reduce == "svd":
        top_k_vectors, top_k_eigen_values = svd(feed_fd_2, k)
        filename = os.path.join(
            file, "SVD-{}-subject-{}-top-{}-type-weights.txt".format(model, args.y, args.k))
    elif args.dim_reduce == "lda":
        # http://pages.cs.wisc.edu/~pradheep/Clust-LDA.pdf
        top_k_vectors = lda(feed_fd_2, k)
        filename = os.path.join(
            file, "LDA-{}-subject-{}-top-{}-type-weights.txt".format(model, args.y, args.k))
    else:
        top_k_vectors = kmeans(feed_fd_2, k)
        filename = os.path.join(
            file, "KMEANS-{}-subject-{}-top-{}-type-weights.txt".format(model, args.y, args.k))
    type_weight_pair = get_subject_weight_pair(k, feed_fd, top_k_vectors)
    save_to_file(type_weight_pair, filename)
    latent_semantics = {
        "latent-type-weight-pair": type_weight_pair,
        "top-k-vectors": top_k_vectors,
        "top-k-eigen-values": top_k_eigen_values,
        "dim-reduce": args.dim_reduce,
        "fd-model": args.fd_model,
        "cm-type": args.cm_type,
        "k": args.k,
        "task": "task-2"
    }
    filename = "{}-latent-semantics.txt".format(filename[:-4])
    save_to_file(latent_semantics, filename)
    filename = "{}.pkl".format(filename[:-4])
    save_to_file(latent_semantics, filename, "pickle")
    return latent_semantics
