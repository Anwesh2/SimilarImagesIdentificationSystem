
from numpy.lib.npyio import save
from utils.utils import pca, svd, lda, kmeans, save_to_file

import os
import numpy as np
import pickle


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

    # Group fd per subject
    feed_fd = {}
    feed_fd_2 = []
    # max = 0
    for idx, i in images.items():
        xyz = idx.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        if x == args.x:
            feed_fd_2.append(list(np.array(i).ravel()))
            if y in feed_fd:
                feed_fd[y].append(list(np.array(i).ravel()))
            else:
                feed_fd[y] = list([np.array(i).ravel()])

    subject_weight_pair = {}
    file = os.path.join(args.image_dir, "phase-2/task-1-output")
    os.makedirs(file, exist_ok=True)
    filename = None

    # Apply Dimension reductionality
    top_k_eigen_values = []
    if args.dim_reduce == "pca":
        # https://medium.com/@devalshah1619/face-recognition-using-eigenfaces-technique-f221d505d4f7
        # paper: http://www.mit.edu/~9.54/fall14/Classes/class10/Turk%20Pentland%20Eigenfaces.pdf
        top_k, top_k_eigen_values = pca(feed_fd_2, k)

        filename = os.path.join(
            file, "PCA-{}-{}-top-{}-subject-weights.txt".format(model, args.x, args.k))
    elif args.dim_reduce == "svd":
        top_k, top_k_eigen_values = svd(feed_fd_2, k)
        filename = os.path.join(
            file, "SVD-{}-{}-top-{}-subject-weights.txt".format(model, args.x, args.k))
    elif args.dim_reduce == "lda":
        # http://pages.cs.wisc.edu/~pradheep/Clust-LDA.pdf
        top_k = lda(feed_fd_2, k)
        filename = os.path.join(
            file, "LDA-{}-{}-top-{}-subject-weights.txt".format(model, args.x, args.k))
    else:
        top_k = kmeans(feed_fd_2, k)
        filename = os.path.join(
            file, "KMEANS-{}-{}-top-{}-subject-weights.txt".format(model, args.x, args.k))

    subject_weight_pair = get_subject_weight_pair(k, feed_fd, top_k)
    save_to_file(subject_weight_pair, filename)
    latent_semantics = {
        "latent-subject-weight-pair": subject_weight_pair,
        "top-k-vectors": top_k,
        "top-k-eigen-values": top_k_eigen_values,
        "dim-reduce": args.dim_reduce,
        "fd-model": args.fd_model,
        "cm-type": args.cm_type,
        "k": args.k,
        "task": "task-1"
    }
    filename = "{}-latent-semantics.txt".format(filename[:-4])
    save_to_file(latent_semantics, filename)
    filename = "{}.pkl".format(filename[:-4])
    save_to_file(latent_semantics, filename, "pickle")
    return latent_semantics


def get_subject_weight_pair(k, feed_fd, top_k):
    subject_weight_pair = {}
    weight_list = {}
    for i in range(k):
        for subject in feed_fd:
            weight_per_subject = 0
            for subject_fd in feed_fd[subject]:
                weight_per_subject = weight_per_subject + \
                    np.matmul(subject_fd, top_k[:, i])
            weight_list[subject] = weight_per_subject
        sum = np.sum(list(weight_list.values()))
        for subject in weight_list:
            weight_list[subject] = weight_list[subject] / sum
        subject_weight_pair["{}th-latent".format(i+1)] = dict(
            sorted(weight_list.items(), key=lambda item: item[1], reverse=True))
    return subject_weight_pair
