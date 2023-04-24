from utils.utils import pca, svd, lda, kmeans, save_to_file

import os
import numpy as np
import pickle
from PIL import Image
import utils
#from utils.utils import color_moment, extended_local_binary_pattern, histogram_oriented_gradient
from tasks import task1, task2, task4, task3
import argparse
import numpy as np
import os
import pickle
# import cv2 as cv
import math
import numpy as np
import sys
import glob
import operator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import gridspec


def Insert(DictMapping, element1):
    counter = len(DictMapping)
    relation = int(element1)
    if relation not in DictMapping:
        DictMapping[relation] = counter
        counter += 1
    return relation


def find_distance_score(element, data_matrix):
    return np.sqrt(np.sum(np.square(data_matrix - element), axis=1))


def VisualizeSimilarImages(ResultImagePathsList, file):
    plots = len(ResultImagePathsList)
    number_cols = 3
    number_rows = int(math.ceil(plots / number_cols))

    gs = gridspec.GridSpec(number_rows, number_cols)
    fig = plt.figure()
    image_paths = ResultImagePathsList
    for x in range(plots):
        ax = fig.add_subplot(gs[x])
        image_path = ResultImagePathsList[x]
        ax.imshow(plt.imread(image_path), cmap='gray')
        ax.axis('off')
    plt.savefig(file + '/topsimilarimages.png')
    plt.show()


def similarity_graph_compute(similarity_matrix, k):
    Subjects = l = [i for i in range(1, 41)]
    ImageGraph = {}
    shape = similarity_matrix.shape[0]
    for element in range(shape):
        present_subject = similarity_matrix[element]
        present_subject_id = Subjects[element]
        DistanceResults = find_distance_score(
            present_subject, similarity_matrix)
        ImageDistanceMapping = {x: y for x, y in zip(
            range(len(Subjects)), DistanceResults)}
        ImageDistanceResults = sorted(
            ImageDistanceMapping.items(), key=lambda x: x[1])[:k]
        ImageGraph[element] = [x for x, y in ImageDistanceResults]
    return ImageGraph


def PageRankCompute(PV, SeedV, matrix):
    beta = 0.85
    epsilon = 0.000001
    for i in range(100):
        ProbV = np.copy(PV)
        PV = beta * (matrix.dot(ProbV)) + (1 - beta)*SeedV
        Delta = PV - ProbV
        Difference = 0.0
        for i in range(len(Delta)):
            Difference = Delta[i] * Delta[i]
        Difference = np.sum(Difference, axis=0)
        Difference = Difference ** 0.5
        if Difference < epsilon:
            break
    return PV


def PersonalizedPageRank(similarity_matrix, subjectid1, subjectid2, subjectid3, NoOfSigSubjects, imagesfolder, file):

    # Initialize Matrix
    DictMapping = dict()
    # for i in range(0, 40):
    for i in range(1, 41):
        Insert(DictMapping, i)

    NormalizedMatrix = similarity_matrix / np.sum(similarity_matrix, axis=0)

    np.nan_to_num(NormalizedMatrix)
    NormalizedMatrix[np.isnan(NormalizedMatrix)] = 0

    SignificantSubjects = [subjectid1, subjectid2, subjectid3]
    seedSize = np.shape(NormalizedMatrix)[0]

    ProbResults = []
    for i in range(seedSize):
        ProbResults.append(1.0 / float(seedSize))

    SS = np.zeros(seedSize)
    for i in SignificantSubjects:
        # 3.0 because number of seed images is 3
        SS[DictMapping[i]] = 1.0 / 3.0
    ProbResults = np.transpose(np.array(ProbResults))
    SS = np.transpose(SS)

    PageRank = PageRankCompute(ProbResults, SS, NormalizedMatrix)

    PI = dict()
    Sum = dict()

    for element in SignificantSubjects:
        s = np.zeros(seedSize)
        s[DictMapping[element]] = 1.0
        s = np.transpose(SS)
        PI[element] = PageRankCompute(ProbResults, SS, NormalizedMatrix)

    for element in SignificantSubjects:
        Sum[element] = 0
        for element2 in SignificantSubjects:
            Sum[element] += PI[element][DictMapping[element2]]

    Centrality = Sum[max(Sum.items(), key=operator.itemgetter(1))[0]]

    S_crit = []
    for key in Sum.keys():
        if Centrality - Sum[key] < 0.00001:
            S_crit.append(key)

    PPR = np.zeros(seedSize)
    for element in S_crit:
        PPR = np.add(PPR, PI[element])
    PPR = PPR/len(S_crit)

    ID = PPR.argsort()[::-1]
    PPR = PPR[ID]
    counter = 0
    results = []

    # for i in range(0, int(NoOfSigSubjects)):
    # for i in range(1, int(NoOfSigSubjects)+1):
    #     variable = (list(DictMapping.keys())[
    #                 list(DictMapping.values()).index(ID[i])])
    #     if variable in SignificantSubjects:

    #     results.append(variable)
    #     print('{} - {}'.format(variable, PPR[i]))
    #     counter += 1

    for i in range(1, 41):
        variable = (list(DictMapping.keys())[
                    list(DictMapping.values()).index(ID[i])])
        if variable not in SignificantSubjects:
            results.append(variable)
            counter += 1
            print('{} - {}'.format(variable, PPR[i]))
        elif variable in SignificantSubjects:
            pass
        if counter >= NoOfSigSubjects:
            break

    ResultImagePathsList = []
    for imageid in results:
        #imageid = str(imageid+1)
        imageid = str(imageid)
        image_path = os.path.join(
            imagesfolder, r'image-original-' + imageid + r'-1.png')
        ResultImagePathsList.append(image_path)
    print(ResultImagePathsList)

    VisualizeSimilarImages(ResultImagePathsList, file)


def run(args):
    path = args.path1
    imagesfolder = args.path2
    n = args.n
    NoOfSigSubjects = args.m
    subjectid1 = args.subjectid1
    subjectid2 = args.subjectid2
    subjectid3 = args.subjectid3
    similarity_matrix = np.loadtxt(path, delimiter=",")
    outgoing_img_graph = similarity_graph_compute(similarity_matrix, n)
    file = os.path.join(args.image_dir, "task-9-output")
    os.makedirs(file, exist_ok=True)
    filename = os.path.join(file, "subject-subject-similarity-graph.txt")
    with open(filename, 'w') as f:
        f.write(str(outgoing_img_graph))
    PersonalizedPageRank(similarity_matrix, subjectid1, subjectid2,
                         subjectid3, NoOfSigSubjects, imagesfolder, file)
