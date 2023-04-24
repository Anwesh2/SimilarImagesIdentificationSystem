import math
import sys
import bisect
import numpy as np
import copy
import os
import pickle
from Levenshtein import distance as levenshtein_distance

from numpy.core.fromnumeric import sort
from p2_tasks import task1, task4, phase1, task5
from utils.utils import get_image_from_dir, euclidean, cosine


def generateHashCode(vector, hyperPlanes):
    bools = (np.dot(vector, hyperPlanes.T) > 0).astype('int')
    return ''.join(bools.astype('str'))


def lsh_indexing(query_vector, image_vectors_map, l, k, t, distance_fun, lsh_index={}):
    buckets_accessed = 0
    total_buckets = 0
    images_considered = 0

    query_vector_ravelled = np.ravel(query_vector)
    dimensions = len(query_vector_ravelled)
    imageKeysList = image_vectors_map.items()
    queryHashes = []
    if not lsh_index:
        print ("LSH Indexing")
        Layers = []    
        hyperPlanes = []
        for i in range(l):
            hyperPlanes.append(np.random.randn(k, dimensions))
            queryHashes.append(generateHashCode(query_vector_ravelled, hyperPlanes[i]))
            Layers.append({})
            dict = Layers[i]
            for imageTuple in imageKeysList:
                hash = generateHashCode(np.ravel(imageTuple[1]), hyperPlanes[i])
                # print ("hash", hash)
                if hash not in dict:
                    dict[hash] = []
                    total_buckets += 1
                dict[hash].append(imageTuple)
        lsh_index["Layers"] = Layers
        lsh_index["hyperPlanes"] = hyperPlanes
    else:
        for i in range(l):
            queryHashes.append(generateHashCode(query_vector_ravelled, lsh_index["hyperPlanes"][i]))
    shortListedImages = set()    
    Layers = lsh_index["Layers"]
    hyperPlanes = lsh_index["hyperPlanes"]
    for i in range(len(Layers)):
        dict = Layers[i]
        queryHash = queryHashes[i]
        keys = dict.keys()
        
        if queryHash in dict:
            buckets_accessed += 1
            bucketImages = dict[queryHash]
            for imageTuple in bucketImages:
                shortListedImages.add(imageTuple[0])
                images_considered += 1
    
    # If number of images is less than required
    distance = 1
    while (len(shortListedImages) < t):
        # print ("distance", distance)
        for i in range (len(Layers)):
            dict = Layers[i]
            queryHash = queryHashes[i]
            keys = dict.keys()
            for key in keys:
                if levenshtein_distance(key, queryHash) <= distance and (levenshtein_distance(key, queryHash) > 0):
                    bucketImages = dict[key]
                    buckets_accessed += 1

                    for imageTuple in bucketImages:
                        shortListedImages.add(imageTuple[0])
                        images_considered += 1
        distance += 1
        
    
    numberOfUniqueImages = len(shortListedImages)
    imageKeysList = list(shortListedImages)
    dist = []
    for image in imageKeysList:
        dist.append(distance_fun(query_vector_ravelled, np.ravel(image_vectors_map[image])))

    # print ("dist", dist)
    sortedDist = np.array(dist).argsort()
    # print ("sortedDist", sortedDist)
    top_t_images = []
    top_t_dist = []
    for idx in range(t):
        # print (sortedDist[idx])
        top_t_images.append(imageKeysList[sortedDist[idx]])     
        top_t_dist.append(dist[sortedDist[idx]])
    
    # print ("topKImages", imageKeysList)
    index_size = sys.getsizeof(Layers)
    # top_t_images, top_t_dist, index_size, buckets_accessed, total_buckets, images_considered
    return [top_t_images, top_t_dist, index_size, buckets_accessed, total_buckets, images_considered, numberOfUniqueImages, lsh_index]

def distance_fun(a, b):
    # return cosine(a, b)
    return euclidean(a, b)

def preprocess(args):
    phase1.task2(args)

def run(args, print_flag=True, index={}):
    # main()
    # return
    # Preprocess to generate FD for image directory
    args.image_dir = args.image_dir_path
    args_copy = copy.deepcopy(args)
    preprocess(args_copy)

    # Load fd of image dir
    # Find model and load corresponding feature descriptor
    path1_images_fd = {}
    model = args.fd_model
    task2_out = os.path.join(args.image_dir_path,
                            "fd", model)
    if args.fd_model == "cm":
        task2_out = os.path.join(task2_out, args.cm_type)
        model = "{}-{}".format(model, args.cm_type)
    task2_out = os.path.join(task2_out, "fd.pkl")
    with open(task2_out, "rb") as f:
        path1_images_fd = pickle.load(f)

    query_image_fd = None
    # if args.query_image_file not in images:
    test_image = get_image_from_dir(os.path.join(args.query_image_file))
    _, fd = phase1.get_feature_descriptor(
        args, test_image)
    if args.fd_model == "hog":
        query_image_fd = fd[1]
    else:
        query_image_fd = fd

    # If Latent semantic file is given we should first transfer each of images fd
    # into that latent sematic before doing LSH file indexing
    if args.latent_semantic_file is not None:
        data = {}
        with open(args.latent_semantic_file, "rb") as f:
            data = pickle.load(f)
        latent_semantics = data["top-k-vectors"]
        if data["task"] == "task-3" or data["task"] == "task-4":
            query_image_fd = task5.transform_to_latent_space(task5.get_nd_of_image(
                query_image_fd, data, path1_images_fd), latent_semantics)
        else:
            query_image_fd = np.matmul(
                    np.asarray(query_image_fd).ravel(), latent_semantics)
        path1_images_fd_copy = copy.deepcopy(path1_images_fd)
        for img_name in path1_images_fd:
            xyz = img_name.split(".")[0].split("-")
            x, y, z = xyz[1], xyz[2], xyz[3]
            if data["task"] == "task-3" or data["task"] == "task-4":
                if img_name in data["nd-data"]:
                    path1_images_fd[img_name] = task5.transform_to_latent_space(
                        data["nd-data"][img_name], latent_semantics)
                else:
                    path1_images_fd[img_name] = task5.transform_to_latent_space(task5.get_nd_of_image(
                        path1_images_fd[img_name], data, path1_images_fd_copy), latent_semantics)
            else:
                current_image_fd = np.nan_to_num(
                    path1_images_fd[img_name].ravel(), posinf=0, neginf=0)
                current_image_in_latent_space = np.matmul(
                    current_image_fd, latent_semantics)
                current_image_in_latent_space = np.nan_to_num(
                    current_image_in_latent_space)
                path1_images_fd[img_name] = current_image_in_latent_space
    top_t_images, top_t_dist, index_size, buckets_accessed, total_buckets, images_considered, unique_images, lsh_index = lsh_indexing(query_image_fd, path1_images_fd, args.l, args.k, args.t, distance_fun, lsh_index=index)
    table_data = []
    for img, dist in zip(top_t_images, top_t_dist):
        table_data.append([img, dist])
    if print_flag:
        print("LSH FILE TOP-T IMAGES")
        print("=====================================================")
        print("{:<25}| {:<20}|".format("IMAGE", "DISTANCE"))
        print("=====================================================")
        for row in table_data:
            print("{:<25}| {:<20.3f}|".format(*row))
    # args.k = args.t + 1 because phase1 task3 includes query image also into account.
    args.k = args.t + 1
    args.images = path1_images_fd
    if args.query_image_file not in args.images:
        args.images[args.query_image_file] = query_image_fd
    if print_flag:
        print("\nTRUE TOP-T IMAGES")
    faces, top_t_expected, top_t_dist_expected, original_faces = phase1.task3(args, print_flag)
    FP = set(top_t_images) - set(top_t_expected[1:])
    FN = set(top_t_expected[1:]) - set(top_t_images)
    TN = (set(path1_images_fd.keys()) - set(top_t_expected[1:])) - set(top_t_images)
    TP = set(top_t_expected[1:]) & set(top_t_images)

    if print_flag:
        print("\nBuckets accessed: " + str(buckets_accessed) + " out of " + str(total_buckets))
        print("LSH FILE INDEX SIZE: ", index_size, " bytes")
        print("Overall images considered: ", images_considered, " out of ", len(path1_images_fd))
        print("Unique images considered: ", unique_images, " out of ", len(path1_images_fd))
        
        print("\nFP: ", FP)
        print("FN: ", FN)

        print("\nFP: {}, FN: {}, TN: {}, TP: {}\n".format(len(FP), len(FN), len(TN), len(TP)))
        print("FALSE POSITIVE RATE: ", (unique_images-len(TP))/args.t, "w.r.t the size of the target set")
        print("FALSE POSITIVE RATE: ", (unique_images-len(TP))/unique_images, "w.r.t the candidate set generated")
        print("MISS RATE: ", (args.t-len(TP))/args.t)
        print("ACCURACY: ", (len(TP)+len(TN))/len(path1_images_fd))
        original_faces = [get_image_from_dir(os.path.join(args.query_image_file))]
        top_t_dist.insert(0, 0)
        for i, img in enumerate(top_t_images):
            top_t_images[i] = os.path.join(args.image_dir, img)
        top_t_images.insert(0, args.query_image_file)
        count = 25
        for img in top_t_images:
            original_faces.append(get_image_from_dir(
                img))
            count -= 1
            if count == 1:
                break
        phase1.show_olivetti_faces(original_faces, top_t_images, top_t_dist,
                            "top n images")()
    return top_t_images, top_t_dist, path1_images_fd, query_image_fd, lsh_index
