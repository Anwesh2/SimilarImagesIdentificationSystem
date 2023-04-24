from PIL import Image
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from utils.utils import get_image_from_dir, color_moment, extended_local_binary_pattern, histogram_oriented_gradient, FDM, save_to_file


def show_olivetti_faces(faces, ind, dist, title):
    """
    Displays Olivette faces
    """
    fig1 = plt.figure(title, figsize=(16, 8))
    plt.subplots_adjust(top=0.9, bottom=0.01, hspace=1,
                        wspace=0)
    f_index1 = 1
    for face, i, d in zip(faces, ind, dist):
        img_grid1 = fig1.add_subplot(5, 5, f_index1)
        f_index1 = f_index1 + 1
        img_grid1.imshow(face, cmap='gray')
        img_grid1.set_title(
            "ID: {}\nDist: {:2.2f}".format(i, d), fontsize=6)
    return plt.show


def display_feature_descriptor_comparison(image, fd, path=None, model=None):
    """Displays feature descriptor of given image and actual image side by side

    Args:
        image (ndarray): Actual image 64*64
        fd (ndarray): feature descriptor of image
        path (str, optional): Path if provided will store at specified location or else just display. Defaults to None.
    """
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image, cmap='gray')
    if model is not None and model == "hog":
        ax2.imshow(fd[1], cmap='gray')
    else:
        ax2.imshow(fd, cmap='gray')
    if not path:
        plt.show()
    else:
        plt.savefig(path)
        if model is not None and model == "hog":
            np.savetxt("{}.txt".format(path),
                       fd[0], fmt="%.2f", delimiter=", ")
        else:
            np.savetxt("{}.txt".format(path), fd, fmt="%.2f", delimiter=", ")
    plt.close()


def get_feature_descriptor(args, image):
    """Fetches Olivetti data set and processes desired image to get feature descriptor

    Args:
        args (Namespace): Object containing all args passed by user
        image (ndarray): Image pixels 64* 64

    Returns:
        [(ndarray, ndarray)]: Returns processed image and corresponding feature decriptor
    """
    cm_dict = {"mean": 1, "std": 2, "skew": 3}
    cm_type = cm_dict[args.cm_type]
    fdm_dict = {"cm": 1, "elbp": 2, "hog": 3}
    feature_descriptor_model = fdm_dict[args.fd_model]
    fd = None
    if feature_descriptor_model == FDM.CM:
        fd = color_moment(image=image, cm_type=cm_type)
    elif feature_descriptor_model == FDM.ELBP:
        fd = extended_local_binary_pattern(image=image)
    else:
        fd, image_fd = histogram_oriented_gradient(image=image)
        fd = [image_fd, fd]
    return image, fd


def task2(args):
    """Given a folder with images, extracts and stores feature descriptors for all the images in the folder with name "fd" for each of model(CM, ELBP, HOG). folder: user-test-image-dir/fd/elbp

    Note1: feature descriptors of each model are stored as pickle file and will be reused later by loading just pickle file.
    folder: user-test-image-dir/fd/elbp/fd.pkl

    Args:
        args (Namespace): Object containing all args passed by user
    """
    image_dir = args.image_dir
    images = [i for i in os.listdir(
        image_dir) if os.path.isfile(os.path.join(image_dir, i))]
    if os.path.exists(os.path.join(image_dir, "fd")):
        return

    def store_fd(args, path):
        fd_data = dict()
        nonlocal images, image_dir
        fp = os.path.join(image_dir, "fd", path)
        for i in images:
            if str.endswith(i, "DS_Store"):
                continue
            image = Image.open(os.path.join(image_dir, i))
            image = np.array(image)
            image = np.divide(image, 255)
            image, fd = get_feature_descriptor(args, image)
            if args.fd_model == "hog":
                fd_data[i] = fd[1]
            else:
                fd_data[i] = fd
        with open(os.path.join(fp, "fd.pkl"), "wb+") as f:
            fd_data = np.nan_to_num(fd_data, neginf=0, posinf=0)
            pickle.dump(fd_data, f)
    # Create directory structure for storing
    os.makedirs(os.path.join(image_dir, "fd"), exist_ok=True)
    os.makedirs(os.path.join(image_dir, "fd", "cm"), exist_ok=True)
    os.makedirs(os.path.join(image_dir, "fd",
                "cm", "mean", "images"), exist_ok=True)
    os.makedirs(os.path.join(image_dir, "fd",
                "cm", "std", "images"), exist_ok=True)
    os.makedirs(os.path.join(image_dir, "fd",
                "cm", "skew", "images"), exist_ok=True)
    os.makedirs(os.path.join(
        image_dir, "fd", "elbp", "images"), exist_ok=True)
    os.makedirs(os.path.join(image_dir, "fd",
                "hog", "images"), exist_ok=True)
    args.fd_model = "cm"
    args.cm_type = "mean"
    store_fd(args, "cm/mean")
    args.fd_model = "cm"
    args.cm_type = "std"
    store_fd(args, "cm/std")
    args.fd_model = "cm"
    args.cm_type = "skew"
    store_fd(args, "cm/skew")
    args.fd_model = "elbp"
    store_fd(args, "elbp")
    args.fd_model = "hog"
    store_fd(args, "hog")


def task3(args):
    """Given a folder with images and an image ID, a model, and a value “k”, returns and visualizes the most similar k images based on the corresponding visual descriptors. For each match, also lists the overall matching score.

    Args:
        args (Namespace): Object containing all args passed by user

    Returns:
        ndarray, list, list, ndarray: feature descriptor of top k images, image name list, distance scores from given query image, actual top k images all in order
    """
    k = args.k
    feed_fd = []
    model = args.fd_model
    task2_out = os.path.join(args.image_dir, "fd", model)
    if args.fd_model == "cm":
        task2_out = os.path.join(task2_out, args.cm_type)
        model = "{}/{}".format(model, args.cm_type)
    print("\n============MODEL: {}============".format(model))
    task2_out = os.path.join(task2_out, "fd.pkl")
    if not os.path.isfile(task2_out):
        task2(args)
    images = args.images
    if images is None:
        with open(task2_out, "rb") as f:
            images = pickle.load(f)

    index_to_image = dict()
    for idx, i in enumerate(images):
        index_to_image[idx] = i
        feed_fd.append(images[i].ravel())
    feed_fd = np.array(feed_fd)

    # Paper that explains how distance metric affect KNN accuracy
    # https://arxiv.org/pdf/1708.04321.pdf
    # if model == "cm/skew":
    #     metric = "minkowski"
    # else:
    #     metric = wasserstein
    metric = "minkowski"
    neigh = NearestNeighbors(n_neighbors=k, radius=1000.0, algorithm='ball_tree',
                             leaf_size=400, metric=metric, p=1, metric_params=None, n_jobs=-1)
    try:
        feed_fd = np.nan_to_num(feed_fd, neginf=0, posinf=0)
        neigh.fit(feed_fd)
    except Exception as ex:
        # print(args)
        print("EXCEPTION in FIT")
        np.set_printoptions(threshold=np.inf)
        # print(list(feed_fd))
        with open("temp_logs.txt", "w+") as f:
            f.write(str(ex))
        print(np.any(np.isnan(feed_fd)),
              np.all(np.isfinite(feed_fd)), np.where(
                  np.isnan(feed_fd)), np.where(np.isinf(feed_fd)))
        np.nan_to_num(feed_fd, neginf=0, posinf=0)
        exit()

    input_query = np.array([images[args.query_image_file].ravel()])
    input_query = np.nan_to_num(input_query, neginf=0, posinf=0)
    # USing KNN we find k similar images to input query image
    neigh_dist, neigh_ind = neigh.kneighbors(
        input_query, n_neighbors=k, return_distance=True)
    faces = list()
    original_faces = list()
    for i in neigh_ind[0]:
        faces.append(images[index_to_image[i]])
        original_faces.append(list(get_image_from_dir(
            os.path.join(args.image_dir, index_to_image[i]))))
    image_list = list()
    print("\nIMAGE    :   OVERALL-SCORE    :    DISTANCE-SCORE")
    print("------------------------------------------------------")
    score = k
    for x, y in zip(neigh_ind[0], neigh_dist[0]):
        # if(score >= k-10):
        print("{}   :   {:2.2f}    :    {:2.2f}".format(
            index_to_image[x], score, y))
        image_list.append(index_to_image[x])
        score -= 1
    return faces, image_list, neigh_dist[0], original_faces


def rank_images(args, store, rank, model_rank, weight):
    """Ranks Images by updating data structures passed

    Args:
        args (Namespace): Object containing all args passed by user
        store (dict): Dictionary storing KNN results for each image
        rank (dict): map of image to its overall rank
        model_rank (dict): map of image to its model specific rank
        weight (float): weight or tuning factor for each model in range of [0,1]
    """
    faces, image_list, neigh_dist, original_faces = task3(
        args)
    ind = args.k
    for face, img, dist, org_img in zip(faces, image_list, neigh_dist, original_faces):
        store[img] = [face, dist, org_img]
        score = (ind * weight)
        if img in rank:
            rank[img] += score
        else:
            rank[img] = score
        model_rank[img] = score
        ind -= 1


def task4(args):
    """ Given a folder with images and an image ID and a value “k”, returns and visualizes the most similar k images based on all corresponding visual descriptors. For each match, also lists the overall matching score and the contributions of the individual visual models

    Args:
        args (Namespace): Object containing all args passed by user
    """
    image_dir = args.image_dir
    images = [i for i in os.listdir(
        image_dir) if os.path.isfile(os.path.join(image_dir, i))]
    k = args.k
    rank = dict()
    args.k = len(images)
    args.fd_model = "cm"
    args.cm_type = "mean"
    cm_mean_store = dict()
    cm_mean_rank = dict()
    weight = 0.2  # 10%
    rank_images(args, cm_mean_store, rank, cm_mean_rank, weight)

    args.cm_type = "std"
    cm_std_store = dict()
    cm_std_rank = dict()
    weight = 0.2  # 10%
    rank_images(args, cm_std_store, rank, cm_std_rank, weight)

    args.cm_type = "skew"
    cm_skew_store = dict()
    cm_skew_rank = dict()
    weight = 0.1  # 5%
    rank_images(args, cm_skew_store, rank, cm_skew_rank, weight)

    args.fd_model = "elbp"
    elbp_store = dict()
    elbp_rank = dict()
    weight = 0.5  # 25%
    rank_images(args, elbp_store, rank, elbp_rank, weight)

    args.fd_model = "hog"
    hog_store = dict()
    hog_rank = dict()
    weight = 1  # 50%
    rank_images(args, hog_store, rank, hog_rank, weight)

    # Sort based on image rank score
    sorted_rank = dict(
        sorted(rank.items(), key=lambda item: item[1], reverse=True))
    print("\nTOP K IMAGES:")
    print("IMAGE-NAME       OVERALL-SCORE       CM-MEAN(%)        CM-STD(%)        CM-SKEW(%)        ELBP(%)       HOG(%)")
    print("------------------------------------------------------------------------------------------------------------------------")
    ind = k
    top_k_image_names = list()
    top_k_images = list()
    top_k_distances = list()
    top_k_cm_mean_images = list()
    top_k_cm_mean_dist = list()
    top_k_cm_std_images = list()
    top_k_cm_std_dist = list()
    top_k_cm_skew_images = list()
    top_k_cm_skew_dist = list()
    top_k_elbp_images = list()
    top_k_elbp_dist = list()
    top_k_hog_images = list()
    top_k_hog_dist = list()
    for img in sorted_rank:
        top_k_image_names.append(img)
        top_k_images.append(list(get_image_from_dir(
            os.path.join(args.image_dir, img))))
        top_k_distances.append(hog_store.get(img, [0, 0])[1])
        top_k_cm_mean_images.append(
            cm_mean_store.get(img, [np.array([[]])])[0])
        top_k_cm_mean_dist.append(cm_std_store.get(img, [0, 0])[1])
        top_k_cm_std_images.append(
            cm_std_store.get(img, [np.array([[]])])[0])
        top_k_cm_std_dist.append(cm_std_store.get(img, [0, 0])[1])
        top_k_cm_skew_images.append(
            cm_skew_store.get(img, [np.array([[]])])[0])
        top_k_cm_skew_dist.append(cm_skew_store.get(img, [0, 0])[1])
        top_k_elbp_images.append(
            elbp_store.get(img, [np.array([[]])])[0])
        top_k_elbp_dist.append(elbp_store.get(img, [0, 0])[1])
        top_k_hog_images.append(
            hog_store.get(img, [np.array([[]])])[0])
        top_k_hog_dist.append(hog_store.get(img, [0, 0])[1])
        r = rank.get(img, 0)
        print("{}       {:2.2f}       {:2.2f}({:2.2f})      {:2.2f}({:2.2f})      {:2.2f}({:2.2f})    {:2.2f}({:2.2f})    {:2.2f}({:2.2f})".format(
            img, r, cm_mean_store.get(img, [0, 0])[1], ((cm_mean_rank.get(img, 0)/r)*100), cm_std_store.get(img, [0, 0])[1], ((cm_std_rank.get(img, 0)/r)*100), cm_skew_store.get(img, [0, 0])[1], ((cm_skew_rank.get(img, 0)/r)*100), elbp_store.get(img, [0, 0])[1], ((elbp_rank.get(img, 0))*100/r), hog_store.get(img, [0, 0])[1], ((hog_rank.get(img, 0))*100/r)))
        if(ind == 1):
            break
        ind -= 1
    top_k_images.extend(top_k_cm_mean_images)
    top_k_image_names.extend(top_k_image_names)
    top_k_distances.extend(top_k_cm_mean_dist)
    top_k_images.extend(top_k_cm_std_images)
    top_k_image_names.extend(top_k_image_names)
    top_k_distances.extend(top_k_cm_std_dist)
    top_k_images.extend(top_k_cm_skew_images)
    top_k_image_names.extend(top_k_image_names)
    top_k_distances.extend(top_k_cm_skew_dist)
    top_k_images.extend(top_k_elbp_images)
    top_k_image_names.extend(top_k_image_names)
    top_k_distances.extend(top_k_elbp_dist)
    top_k_images.extend(top_k_hog_images)
    top_k_image_names.extend(top_k_image_names)
    top_k_distances.extend(top_k_hog_dist)
    show_olivetti_faces(top_k_images, top_k_image_names,
                        top_k_distances, "COMBINED TOP K IMAGES")()
