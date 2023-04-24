from utils.utils import pca, svd, lda, kmeans, save_to_file, euclidean

import os
import numpy as np
import pickle


def run(args):
    if args.k > 12:
        print("K can be at max 12")
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

    # Group fd per type
    feed_fd = {}
    feed_fd_2 = {}
    for idx, i in images.items():
        xyz = idx.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        if x in feed_fd:
            feed_fd[x].extend(list(np.array(i).ravel()))
        else:
            feed_fd[x] = list(np.array(i).ravel())
        if x in feed_fd_2:
            feed_fd_2[x].append(np.array(i))
        else:
            feed_fd_2[x] = [np.array(i)]

    file = os.path.join(args.image_dir, "phase-2/task-3-output")
    os.makedirs(file, exist_ok=True)

    similarity_on_whole_data_nd = {}
    task_3_out = os.path.join(
        args.image_dir, "phase-2/task-3-output", "nd_{}.pkl".format(model))
    if os.path.isfile(task_3_out):
        with open(task_3_out, "rb") as f:
            similarity_on_whole_data_nd = pickle.load(f)
    else:
        for idx, i in images.items():
            xyz = idx.split(".")[0].split("-")
            x, y, z = xyz[1], xyz[2], xyz[3]
            similarity_on_whole_data_nd[idx] = []
            for type in feed_fd_2:
                score = 0
                for type_fd in feed_fd_2[type]:
                    score = score + euclidean(np.array(i), type_fd)
                score = score / len(feed_fd_2[type])
                similarity_on_whole_data_nd[idx].append(score)
        save_to_file(similarity_on_whole_data_nd, task_3_out, format="pickle")
    type_list = feed_fd.keys()
    all_type_images = feed_fd.values()

    filename = None

    # Generate type-type similarity matrix
    type_type = []
    # TODO: Can be optimized but as long as it runs without much delay its fine
    for img1 in all_type_images:
        sim_score = []
        for img2 in all_type_images:
            if len(img1) < len(img2):
                img2 = img2[:len(img1)]
            if len(img2) < len(img1):
                img1 = img1[:len(img2)]
            # TODO: Change this to euclidean
            # Why dot instead of p1 or p2?
            # seems almost similar and its easier to calculate dot
            # https://stats.stackexchange.com/questions/544951/when-to-use-dot-product-as-a-similarity-metric
            sim_score.append(euclidean(img1, img2))
        type_type.append(sim_score)

    filename = os.path.join(file, "{}-type-type-similarity.txt".format(model))
    np.savetxt(filename, type_type, fmt="%.6f", delimiter=", ")

    # Apply Dimension reductionality
    top_k_eigen_values = []
    if args.dim_reduce == "pca":
        # https://medium.com/@devalshah1619/face-recognition-using-eigenfaces-technique-f221d505d4f7
        # paper: http://www.mit.edu/~9.54/fall14/Classes/class10/Turk%20Pentland%20Eigenfaces.pdf
        top_k_vectors, top_k_eigen_values = pca(type_type, k)
        filename = os.path.join(
            file, "PCA-{}-top-{}-type-weights.txt".format(model, args.k))
    elif args.dim_reduce == "svd":
        top_k_vectors, top_k_eigen_values = svd(type_type, k)
        filename = os.path.join(
            file, "SVD-{}-top-{}-type-weights.txt".format(model, args.k))
    elif args.dim_reduce == "lda":
        # http://pages.cs.wisc.edu/~pradheep/Clust-LDA.pdf
        top_k_vectors = lda(type_type, k)
        filename = os.path.join(
            file, "LDA-{}-top-{}-type-weights.txt".format(model, args.k))
    else:
        top_k_vectors = kmeans(type_type, k)
        filename = os.path.join(
            file, "KMEANS-{}-top-{}-type-weights.txt".format(model, args.k))
    type_weight_pair = get_type_weight_pair(
        k, type_type, top_k_vectors, type_list)
    save_to_file(type_weight_pair, filename)
    latent_semantics = {
        "latent-type-weight-pair": type_weight_pair,
        "top-k-vectors": top_k_vectors,
        "top-k-eigen-values": top_k_eigen_values,
        "dim-reduce": args.dim_reduce,
        "fd-model": args.fd_model,
        "cm-type": args.cm_type,
        "k": args.k,
        "similarity-matrix": type_type,
        "similarity-list": list(type_list),
        "nd-data": similarity_on_whole_data_nd,
        "task": "task-3"
    }
    filename = "{}-latent-semantics.txt".format(filename[:-4])
    save_to_file(latent_semantics, filename)
    filename = "{}.pkl".format(filename[:-4])
    save_to_file(latent_semantics, filename, "pickle")
    return latent_semantics


def get_type_weight_pair(k, type_type, top_k, type_list):
    type_weight_pair = {}
    weight_list = {}
    for i in range(k):
        for type_sim, t_type in zip(type_type, type_list):
            weight_list[t_type] = np.matmul(type_sim, top_k[:, i])
        sum = np.sum(list(weight_list.values()))
        for type in weight_list:
            weight_list[type] = weight_list[type] / sum
        type_weight_pair["{}th-latent".format(i+1)] = dict(
            sorted(weight_list.items(), key=lambda item: item[1], reverse=True))
    return type_weight_pair
