import math
import sys
import bisect
import numpy as np
import copy
import os
import pickle
from p2_tasks import task1, task4, phase1, task5
from utils.utils import get_image_from_dir, euclidean


def get_bits_for_dimens(b, d):
    bj = []
    for j in range(d):
        if j+1 <= b % d:
            bj.append(math.floor(b/d) + 1)
        else:
            bj.append(math.floor(b/d))
    return bj

def init_candidate(n, dst, ans):
    for k in range(n):
        dst.append(sys.maxsize)
        ans.append(0)
    return sys.maxsize

def sort_on_dst(dst, ans):
    for i in range(len(dst)):
        min = dst[i]
        min_index = i
        for j in range(i, len(dst)):
            if dst[j] < min:
                min = dst[j]
                min_index = j
        dst[i], dst[min_index] = dst[min_index], dst[i]
        ans[i], ans[min_index] = ans[min_index], ans[i]
    return dst, ans

def candidate(d, i, dst, ans):
    if d < dst[len(dst) - 1]:
        dst[len(dst) - 1] = d
        ans[len(dst) - 1] = i
        sort_on_dst(dst, ans)
    return dst[len(dst) - 1]

def compute_p(dimens, vectors, b):
    p = [[]] * dimens
    for dimen in range(dimens):
        entries = []
        for vector in vectors:
            entries.append(vector[dimen])
        entries.sort()
        partition_size = int(len(entries) / pow(2, b[dimen]))
        entries.reverse()
        k = 0
        p_dimen = []
        for entry in entries:
            k = k + 1
            if k >= partition_size:
                p_dimen.append(entry)
                k = 0
        if entries[len(entries) - 1] not in p_dimen:
            p_dimen.append(entries[len(entries) - 1])
        p_dimen.reverse()
        p[dimen] = p_dimen

    # p = [[0,3,9,16,21],[0,5,11]]
    # print(p)
    return p

def find_partition(arr, value):
    for i in range(len(arr)-1, -1, -1):
        if arr[i] <= value:
            return i
    if value < arr[0]:
        return 0
    return len(arr) - 1

def compute_r(v, p):
    r = [[0] * len(v[0]) for _ in range(len(v))]
    for i in range(len(v)):
        for j in range(len(v[i])):
            r[i][j] = find_partition(p[j], v[i][j])
    return r

def compute_r_for_query_vector(query_vector, p):
    r = [None] * len(query_vector)
    for j in range(len(query_vector)):
        r[j] = find_partition(p[j], query_vector[j])
        # print(f"p[{j}]: {p[j]}")
        # print(f"{query_vector[j]}")
        # print(f"r[{j}]: {r[j]}")
    return r

def get_l(query_vector, approx_size, p, r, bct):
    l = [[0] * len(query_vector) for _ in  range(approx_size)]
    rq = compute_r_for_query_vector(query_vector, p)

    for i in range(approx_size):
        for j in range(len(query_vector)):
            # print("r[i][j]: " + str(r[i][j]) + ", rq[j]: " + str(rq[j]))
            if r[i][j] < rq[j]:
                l[i][j] = query_vector[j] - p[j][r[i][j] + 1]
                temp = str(j) + " " + str(r[i][j] + 1)
                bct.add(temp)
            elif r[i][j] == rq[j]:
                l[i][j] = 0
            else:
                l[i][j] = p[j][r[i][j]] - query_vector[j]
                temp = str(j) + " " + str(r[i][j])
                bct.add(temp)
    return l

def get_lower_bounds(l, i, dimen, p=2):
    l_i = 0
    for j in range(dimen):
        l_i += pow(l[i][j], p)
    return pow(l_i, 1/p)

def compute_approximations(vectors, p):
    # a = [""] * len(vectors)
    a = []
    for i in range(len(vectors)):
        r = compute_r_for_query_vector(vectors[i], p)
        a.append(r)
    return a

def p_size(p):
    count = 0
    for i in range(len(p)):
        for j in range(len(p[0])):
            count += 1
    return count

def va_files_approximation(query_vector, image_vectors_map, b, n, distance_fun, va_index={}):
    bct = set()
    image_names = []
    vectors = []
    query_vector = query_vector.ravel()
    for image_name, vector in image_vectors_map.items():
        image_names.append(image_name)
        vectors.append(vector.ravel())
    if not va_index:
        print("VA FILE INDEXING")
        dst = []
        ans = []
        va_index["b"] = get_bits_for_dimens(b, len(query_vector))
        va_index["p"] = compute_p(len(query_vector), vectors, va_index["b"])
        # print("\nNumber of partition points: " + str(p_size(p)))
        va_index["a"] = compute_approximations(vectors, va_index["p"])
        va_index["d"] = init_candidate(n, dst, ans)
        va_index["r"] = compute_r(vectors, va_index["p"])
        va_index["ans"] = ans
        va_index["dst"] = dst
    l = get_l(query_vector, len(va_index["a"]), va_index["p"], va_index["r"], bct)
    images_considered = 0
    for i in range(len(vectors)):
        l_i = get_lower_bounds(l, i, len(va_index["a"][i]))
        if l_i < va_index["d"]:
            images_considered +=1
            va_index["d"] = candidate(distance_fun(vectors[i], query_vector), i, va_index["dst"], va_index["ans"])
    nearest_neighbors = []
    for i in va_index["ans"]:
        nearest_neighbors.append(image_names[i])
    size_of_va_files = sys.getsizeof(va_index["p"])
    return nearest_neighbors, va_index["dst"], size_of_va_files, len(bct), p_size(va_index["p"]), images_considered, va_index

def distance_fun(a, b):
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
    # into that latent sematic before doing VA file approximation
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
    top_t_images, top_t_dist, index_size, buckets_accessed, total_buckets, images_considered, va_index = va_files_approximation(query_image_fd, path1_images_fd, args.b, args.t, distance_fun, va_index=index)
    table_data = []
    for img, dist in zip(top_t_images, top_t_dist):
        table_data.append([img, dist])
    if print_flag:
        print("VA FILE TOP-T IMAGES")
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
        print("VA FILE INDEX SIZE: ", index_size, " bytes")
        print("Overall images considered: ", images_considered, " out of ", len(path1_images_fd))
        print("Unique images considered: ", images_considered, " out of ", len(path1_images_fd))
        
        print("\nFP: ", FP)
        print("FN: ", FN)

        print("\nFP: {}, FN: {}, TN: {}, TP: {}\n".format(len(FP), len(FN), len(TN), len(TP)))
        print("FALSE POSITIVE RATE: ", (images_considered-len(TP))/args.t, "w.r.t the size of the target set")
        print("FALSE POSITIVE RATE: ", (images_considered-len(TP))/images_considered, "w.r.t the candidate set generated")
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
    return top_t_images, top_t_dist, path1_images_fd, query_image_fd, va_index


def main():
    vectors = [
        [1,3],
        [2,3],
        [4, 10],
        [13, 6],
        [18, 1]
    ]
    image_vectors_map = {}
    for i in range(len(vectors)):
        image_vectors_map[str(i)] = vectors[i]
    print(va_files_approximation([20, 3], image_vectors_map, 3, 5, distance_fun))

if __name__ == "__main__":
    main()
