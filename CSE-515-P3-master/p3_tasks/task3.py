import matplotlib.pyplot as plt
from p2_tasks import task1, task4, phase1
from utils.utils import save_to_file, euclidean
from utils.classifier import SVM, display_confusion_matrix, get_miss_rate, svm_predict, get_confusion_matrix, get_fp_rate, get_miss_rate, get_class_count, DT, dt_predict, get_accuracy_score
import copy
import os
import pickle
import numpy as np
import networkx as nx
import time


# 1. Generate latent semantics for given image directory
#   - We can use either phase-2 task-1(i.e subject-weight semantics) or 
#     task-4(i.e subject-subject similarity semantics)
# 2. Transform each of the image to that latent space and
#    train with user specified classifier
# 3. Now transform each of the images into same latent space and
# 4. Use the classifier to classify images into respective types.

# Note: If --dim-reduce argument is not provided we will use original FD of images
def run(args):
    # Preprocess to generate FD for path1 image directory
    args.image_dir = args.path1
    args_copy = copy.deepcopy(args)
    preprocess(args_copy)
    
    # Load fd of path1
    # Find model and load corresponding feature descriptor
    path1_images_fd = {}
    model = args.fd_model
    task2_out = os.path.join(args.path1,
                            "fd", model)
    if args.fd_model == "cm":
        task2_out = os.path.join(task2_out, args.cm_type)
        model = "{}-{}".format(model, args.cm_type)
    task2_out = os.path.join(task2_out, "fd.pkl")
    with open(task2_out, "rb") as f:
        path1_images_fd = pickle.load(f)
    
    # Better to select subject with all type images
    y_x_count = {}
    for image_name, fd in path1_images_fd.items():
        xyz = image_name.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        if x in y_x_count:
            y_x_count[x].add(y)
        else:
            y_x_count[x] = set({y})
    args.x = "original"
    max_x = 0
    for x in y_x_count:
        l = len(y_x_count[x])
        if l > max_x:
            max_x = l
            args.x = x
    print("Using subject {} images for computing latent semantics".format(args.x))
    print("Learnt TYPES: {}".format(y_x_count[args.x]))

    # Get Latent space representing path1 images
    path1_output = os.path.join(args.path1, "task-3-output")
    os.makedirs(path1_output, exist_ok=True)
    latent_file = os.path.join(
        path1_output, "{}-{}-subject-{}-top-{}-type-semantics.pkl".format(args.dim_reduce, model, args.x, args.k))
    data = {}
    if args.dim_reduce is not None:
        if os.path.exists(latent_file):
            with open(latent_file, "rb") as f:
                data = pickle.load(f)
        else:
            data = task1.run(args)
            save_to_file(data, latent_file, "pickle")
        top_k_latent_semantics = data["top-k-vectors"]

    # Transform each of path1 fd images to latent space
    # and build X training data with labels y
    training_data = []
    training_labels = []
    actual_training_labels = []
    for image_name, fd in path1_images_fd.items():
        xyz = image_name.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        training_labels.append("{}-{}".format(y, z))
        actual_training_labels.append(z)
        if args.dim_reduce is not None:
            transformed_fd = transform_fd(np.asarray(fd).ravel(), top_k_latent_semantics)
        else:
            transformed_fd = np.asarray(fd).ravel()
        training_data.append(transformed_fd)
    
    # Load test data i.e path2 images
    # Preprocess to generate FD for path2 image directory
    args.image_dir = args.path2
    args_copy = copy.deepcopy(args)
    preprocess(args_copy)

    # Load fd of path2
    # Find model and load corresponding feature descriptor
    path2_images_fd = {}
    model = args.fd_model
    task2_out = os.path.join(args.path2,
                            "fd", model)
    if args.fd_model == "cm":
        task2_out = os.path.join(task2_out, args.cm_type)
        model = "{}-{}".format(model, args.cm_type)
    task2_out = os.path.join(task2_out, "fd.pkl")
    with open(task2_out, "rb") as f:
        path2_images_fd = pickle.load(f)

    # Transform each of path2 fd images to latent space
    # and build X testing data with expected labels y
    path2_output = os.path.join(args.path2, "task-3-output")
    os.makedirs(path2_output, exist_ok=True)
    testing_data = []
    expected_testing_labels = []
    for image_name, fd in path2_images_fd.items():
        xyz = image_name.split(".")[0].split("-")
        x, y, z = xyz[1], xyz[2], xyz[3]
        expected_testing_labels.append(z)
        if args.dim_reduce is not None:
            transformed_fd = transform_fd(np.asarray(fd).ravel(), top_k_latent_semantics)
        else:
            transformed_fd = np.asarray(fd).ravel()
        testing_data.append(transformed_fd)

    # Train and Classify
    if args.classifier == "svm":
        clf = SVM(training_data, training_labels)
        ypred = svm_predict(clf, testing_data)
        labels = clf.classes_
    elif args.classifier == "dt":
        clf = DT(training_data, training_labels)
        ypred = dt_predict(clf, testing_data)
        labels = clf.classes_
    elif args.classifier == "ppr":
        # print ("training labels", set(training_labels))
        # print (len(training_data), len(training_labels))
        start_time = time.time()
        print ("computing ppr")
        uniqueLabels = list(set(training_labels))
        combinedImages = []
        combinedImages.extend(training_data)
        combinedImages.extend(testing_data)
        graph=nx.DiGraph()
        m = 0
        maxValue = 2147483647
        similarityMatrix = np.zeros(shape=(len(combinedImages), len(combinedImages)))
        print ("computing similarity matrix")
        for i in range (len(combinedImages)):
            for j in range (len(combinedImages)):
                d = euclidean(combinedImages[i], combinedImages[j])
                # graph.add_weighted_edges_from([(i,j,0.5),])
                if (d == 0):
                    inverse = maxValue
                else:
                    inverse = 1 / d
                m = max (m, inverse)     
                # graph.add_edge(i, j, weight=inverse)
                similarityMatrix[i][j] = inverse

        rankMatrix = []
        row, col = len(testing_data), len(uniqueLabels)
        rankMatrix = [[0 for x in range(col)] for y in range(row)]
        # print("--- similarity matrix minutes ---", (time.time() - start_time))
        print ("computing seeds")
        for i in range(len(uniqueLabels)):
            seeds = {}
            start_time = time.time()
            similarityMatrixWithSeeds = similarityMatrix
            for j in range (len(training_data)):
                if (training_labels[j] == uniqueLabels[i]):
                    seeds[j] = m
                    similarityMatrixWithSeeds[j] = similarityMatrixWithSeeds[j] + maxValue

            # print ("computing transition matrix")
            transitionMatrix = similarityMatrixWithSeeds / similarityMatrixWithSeeds.sum(axis=0)
            # print ("computing randomwalk")
            randomWalk = np.linalg.matrix_power(transitionMatrix, 100)
            for k in range (len(randomWalk)):
                if k >= len(training_labels):
                    rankMatrix[k - len(training_labels)][i] = randomWalk[k][0]

            for j in range (len(training_data)):
                if (training_labels[j] == uniqueLabels[i]):
                    seeds[j] = m
                    similarityMatrixWithSeeds[j] = similarityMatrixWithSeeds[j] - maxValue

            # print("--- rank matrix minutes ---", (time.time() - start_time))

        ypred = []
        for i in range(len(rankMatrix)):
            row = rankMatrix[i]
            # print ("row", row)
            max_value = max(row)
            max_index = row.index(max_value)
            ypred.append(uniqueLabels[max_index])
            # print (expected_testing_labels[i], uniqueLabels[max_index])
            labels = uniqueLabels
    
    new_labels = set()
    for l in labels:
        new_labels.add(l.split("-")[1])
    new_labels = list(new_labels)
    new_ypred = []
    for l in ypred:
        new_ypred.append(l.split("-")[1])
    compute_results(args, new_labels, expected_testing_labels, actual_training_labels, new_ypred, path2_images_fd, path2_output, model)

def preprocess(args):
    phase1.task2(args)

def transform_fd(fd, top_k_latent_semantics):
    fd = np.array(fd).ravel()
    top_k_latent_semantics = np.array(top_k_latent_semantics)
    return np.matmul(fd, top_k_latent_semantics)

def compute_results(args, labels, expected_testing_labels, training_labels, ypred, path2_images_fd, path2_output, model):
    cm, FP, FN, TP, TN = get_confusion_matrix(expected_testing_labels, ypred, labels)
    miss_rate = get_miss_rate(FN, TP)
    fp_rate = get_fp_rate(FP, TN)
    train_count = get_class_count(training_labels)
    expected_count = get_class_count(expected_testing_labels)
    print("{:<25}| {:<20}| {:<20}| {:<25}|".format("CLASS(train/test)", "FPR", "MISS-RATE", "FP/FN/TP/TN"))
    print("==========================================================================================")
    table_data = []
    for cls, fpr, mr, _FP, _FN, _TP, _TN in zip(labels, fp_rate, miss_rate, FP, FN, TP, TN):
        train = 0
        if cls in train_count:
            train = train_count[cls]
        expected = 0
        if cls in expected_count:
            expected = expected_count[cls]
        
        table_data.append(["{}({}/{})".format(cls, train, expected), fpr, mr, "{}/{}/{}/{}".format(_FP, _FN, _TP, _TN)])
        # print("{}({}/{})    {:10.3f}    {:10.3f}    {:10}    {:10}    {:10}    {:10}".format(cls, train, expected, fpr, mr, _FP, _FN, _TP, _TN))
    for row in table_data:
        print("{:<25}| {:<20.3f}| {:<20.3f}| {:<25}|".format(*row))
    print("\nAVERAGE FPR: {}".format(np.nansum(fp_rate)/len(fp_rate)))
    print("AVERAGE MISS RATE: {}".format(np.nansum(miss_rate)/len(miss_rate)))
    print("ACCURACY SCORE: {}".format(get_accuracy_score(expected_testing_labels, ypred)))
    display_confusion_matrix(cm, labels)
    prediction = {}
    output = {
        "table-data": table_data,
        "prediction-labels": prediction
    }
    for (image_name, fd), label in zip(path2_images_fd.items(), ypred):
        prediction[image_name] = label
    save_to_file(output, os.path.join(path2_output, "{}-{}-top-{}-{}-output.txt".format(args.classifier, model, args.k, args.dim_reduce)))