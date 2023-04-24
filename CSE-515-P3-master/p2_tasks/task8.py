import numpy as np
import matplotlib.pyplot as plt

from utils import ascos_plus_plus


def run(args):
    input_matrix = np.loadtxt(args.sss_matrix_path, delimiter=",")
    max_similarity = input_matrix.ravel().max()
    for i in range(0, len(input_matrix)):
        for j in range(0, len(input_matrix)):
            input_matrix[i][j] = max_similarity - input_matrix[i][j]
    most_significant_subjects = ascos_plus_plus.compute_similar_using_ascos_plus_plus(
        input_matrix, args.n, args.m, args.original_images_path)
    fig = plt.figure()
    for i in range(len(most_significant_subjects)):
        subject = most_significant_subjects[i]
        subplot = fig.add_subplot(1, args.m, i+1)
        subplot.imshow(plt.imread(args.original_images_path +
                       "/image-original-" + str(int(subject)) + "-1.png"), cmap='gray')
    plt.show()
    print("MOST SIGNIFICANT SUBJECTS ARE: {}".format(most_significant_subjects))
