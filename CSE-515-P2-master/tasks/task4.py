
from utils.utils import pca, svd, lda, kmeans, save_to_file, euclidean

import os
import numpy as np
import pickle
from .task3 import get_type_weight_pair


def run(args):
    if args.k > 40:
        print("K can be at max 40")
        exit()
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
    feed_fd_2 = {}
    for idx, i in images.items():
        xyz = idx.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        # TODO: For now doing it for just original, but it can be updated to
        # calculated for each type and later just take average similarity
        if y in feed_fd:
            feed_fd[y].extend(list(np.array(i).ravel()))
        else:
            feed_fd[y] = list(np.array(i).ravel())
        if y in feed_fd_2:
            feed_fd_2[y].append(np.array(i))
        else:
            feed_fd_2[y] = [np.array(i)]

    feed_fd = dict(
        sorted(feed_fd.items(), key=lambda item: int(item[0])))
    feed_fd_2 = dict(
        sorted(feed_fd_2.items(), key=lambda item: int(item[0])))
    file = os.path.join(args.image_dir, "task-4-output")
    os.makedirs(file, exist_ok=True)
    similarity_on_whole_data_nd = {}
    task_4_out = os.path.join(
        args.image_dir, "task-4-output", "nd_{}.pkl".format(model))
    filename = None
    all_subject_images = feed_fd.values()
    subject_list = feed_fd.keys()
    if os.path.isfile(task_4_out):
        with open(task_4_out, "rb") as f:
            similarity_on_whole_data_nd = pickle.load(f)
    else:
        for idx, i in images.items():
            xyz = idx.split(".")[0].split("-")
            x, y, z = xyz[1], xyz[2], xyz[3]
            similarity_on_whole_data_nd[idx] = []
            for type in subject_list:
                score = 0
                for type_fd in feed_fd_2[type]:
                    score = score + euclidean(np.array(i), type_fd)
                score = score / len(feed_fd_2[type])
                similarity_on_whole_data_nd[idx].append(score)
        save_to_file(similarity_on_whole_data_nd, task_4_out, format="pickle")

    # Generate subject-subject similarity
    subject_subject = []
    # TODO: Can be optimized but as long as it runs without much delay its fine
    for img1 in all_subject_images:
        sim_score = []
        for img2 in all_subject_images:
            if len(img1) < len(img2):
                img2 = img2[:len(img1)]
            if len(img2) < len(img1):
                img1 = img1[:len(img2)]
                # Why dot instead of p1 or p2?
                # seems almost similar and its easier to calculate dot
                # https://stats.stackexchange.com/questions/544951/when-to-use-dot-product-as-a-similarity-metric
            sim_score.append(euclidean(img1, img2))
        subject_subject.append(sim_score)

    filename = os.path.join(
        file, "{}-subject-subject-similarity.txt".format(model))
    np.savetxt(filename, subject_subject, fmt="%.6f", delimiter=", ")

    # Apply Dimension reductionality
    top_k_eigen_values = []
    if args.dim_reduce == "pca":
        # https://medium.com/@devalshah1619/face-recognition-using-eigenfaces-technique-f221d505d4f7
        # paper: http://www.mit.edu/~9.54/fall14/Classes/class10/Turk%20Pentland%20Eigenfaces.pdf
        top_k_vectors, top_k_eigen_values = pca(subject_subject, k)
        filename = os.path.join(
            file, "PCA-{}-top-{}-subject-weights.txt".format(model, args.k))
    elif args.dim_reduce == "svd":
        top_k_vectors, top_k_eigen_values = svd(subject_subject, k)
        filename = os.path.join(
            file, "SVD-{}-top-{}-subject-weights.txt".format(model, args.k))
    elif args.dim_reduce == "lda":
        # http://pages.cs.wisc.edu/~pradheep/Clust-LDA.pdf
        top_k_vectors = lda(subject_subject, k)
        filename = os.path.join(
            file, "LDA-{}-top-{}-subject-weights.txt".format(model, args.k))
    else:
        top_k_vectors = kmeans(subject_subject, k)
        filename = os.path.join(
            file, "KMEANS-{}-top-{}-subject-weights.txt".format(model, args.k))

    subject_weight_pair = get_type_weight_pair(
        k, subject_subject, top_k_vectors, subject_list)
    # for klatent, subject in zip(top_k_vectors, subject_list):
    #     subject_weight_pair[subject] = klatent
    save_to_file(subject_weight_pair, filename)
    latent_semantics = {
        "latent-type-weight-pair": subject_weight_pair,
        "top-k-vectors": top_k_vectors,
        "top-k-eigen-values": top_k_eigen_values,
        "dim-reduce": args.dim_reduce,
        "fd-model": args.fd_model,
        "cm-type": args.cm_type,
        "k": args.k,
        "similarity-matrix": subject_subject,
        "similarity-list": list(subject_list),
        "nd-data": similarity_on_whole_data_nd,
        "task": "task-4"
    }
    filename = "{}-latent-semantics.txt".format(filename[:-4])
    save_to_file(latent_semantics, filename)
    filename = "{}.pkl".format(filename[:-4])
    save_to_file(latent_semantics, filename, "pickle")
