from tasks import task1, task2, task4, task3, task5, phase1, task6, task7, task8, task9
import argparse
import numpy as np
import copy

# Uncomment below line if using windows
# matplotlib.use('')

# Set Numpy print options globally
np.set_printoptions(threshold=np.inf, precision=3, suppress=True)


def parse_args():
    """Argument parser for the script

    Returns:
        Namespace: Object containing all args passed by user
    """
    arg_parser = argparse.ArgumentParser()
    subparsers = arg_parser.add_subparsers(
        help='sub-command help',
        dest='commands')
    subparsers.required = True
    t1_parser = subparsers.add_parser("task-1", help="First task options")
    t1_parser.add_argument("-x", type=str, choices=["cc", "con", "detail", "emboss", "jitter", "neg", "noise01", "noise02",
                           "original", "poster", "rot", "smooth", "stipple"], default="rot", help="Enter image type. (deafult: rot)")
    t1_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k for top k latent semantics. (deafult: 2)")
    t1_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'], help="Enter feature descriptor model from available choices. (default: cm)", type=str, default="cm")
    t1_parser.add_argument(
        "--dim-reduce", choices=['pca', 'svd', 'lda', 'kmeans'], help="Enter dimensionality reduction technique from available choices. (default: pca)", type=str, default="pca")
    t1_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"], help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    t2_parser = subparsers.add_parser("task-2", help="Second task options")
    t2_parser.add_argument("-y", type=int, default=1,
                           help="Enter subject. (deafult: 1)", choices=range(1, 41), metavar="[1-40]")
    t2_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k for top k latent semantics. (deafult: 2)")
    t2_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'], help="Enter feature descriptor model from available choices. (default: cm)", type=str, default="cm")
    t2_parser.add_argument(
        "--dim-reduce", choices=['pca', 'svd', 'lda', 'kmeans'], help="Enter dimensionality reduction technique from available choices. (default: pca)", type=str, default="pca")
    t2_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"], help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    t3_parser = subparsers.add_parser("task-3", help="Third task options")
    t3_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k for top k latent semantics. (deafult: 2)")
    t3_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'], help="Enter feature descriptor model from available choices. (default: cm)", type=str, default="cm")
    t3_parser.add_argument(
        "--dim-reduce", choices=['pca', 'svd', 'lda', 'kmeans'], help="Enter dimensionality reduction technique from available choices. (default: pca)", type=str, default="pca")
    t3_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"], help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    t4_parser = subparsers.add_parser("task-4", help="Fourth task options")
    t4_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k for top k latent semantics. (deafult: 2)")
    t4_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'], help="Enter feature descriptor model from available choices. (default: cm)", type=str, default="cm")
    t4_parser.add_argument(
        "--dim-reduce", choices=['pca', 'svd', 'lda', 'kmeans'], help="Enter dimensionality reduction technique from available choices. (default: pca)", type=str, default="pca")
    t4_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"], help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    t5_parser = subparsers.add_parser("task-5", help="Fifth task options")
    t5_parser.add_argument("--query-image-file", type=str,
                           default="test-image-sets/all/image-original-1-1.png", help="Enter image/s directory path")
    t5_parser.add_argument("--latent-semantic-file", type=str,
                           default="test-image-sets/all/task-1-output/PCA-cm-mean-rot-top-2-subject-weights-latent-semantics.pkl", help="Enter image/s directory path")
    t5_parser.add_argument("-n", type=int, default=3,
                           help="Enter value n for top n similar images. (deafult: 3)")
    t6_parser = subparsers.add_parser("task-6", help="Sixth task options")
    t6_parser.add_argument("--query-image-file", type=str,
                           default="test-image-sets/all/image-original-1-1.png", help="Enter image/s directory path")
    t6_parser.add_argument("--latent-semantic-file", type=str,
                           default="test-image-sets/all/task-1-output/PCA-cm-mean-rot-top-2-subject-weights-latent-semantics.pkl", help="Enter image/s directory path")
    t7_parser = subparsers.add_parser("task-7", help="SEVENTH task options")
    t7_parser.add_argument("--query-image-file", type=str,
                           default="test-image-sets/all/image-original-1-1.png", help="Enter image/s directory path")
    t7_parser.add_argument("--latent-semantic-file", type=str,
                           default="test-image-sets/all/task-1-output/PCA-cm-mean-rot-top-2-subject-weights-latent-semantics.pkl", help="Enter image/s directory path")
    t8_parser = subparsers.add_parser("task-8", help="Eight task options")
    t8_parser.add_argument("--sss_matrix_path", type=str,
                           default="test-image-sets/all/task-4-output/cm-mean-subject-subject-similarity.txt", help="Enter subject subject similarity matrix file path")
    t8_parser.add_argument("--original_images_path", type=str,
                           default="test-image-sets/all/", help="Enter path where original input images are present")
    t8_parser.add_argument("-n", type=int, default=40,
                           help="Enter value n, for computing n most similar objects for subject")
    t8_parser.add_argument("-m", type=int, default=4,
                           help="Enter value m, for finding m most significant subjects")
    t9_parser = subparsers.add_parser("task-9", help="Ninth task options")
    t9_parser.add_argument("-path1", type=str, default="test-image-sets/all/task-4-output/cm-mean-subject-subject-similarity.txt",
                           help="Enter file path for subject-subject similarity matrix")
    t9_parser.add_argument("-path2", type=str, default="test-image-sets/all",
                           help="Enter file path for dataset of all images")
    t9_parser.add_argument("-n", type=int, default=2,
                           help="Enter value of n for n most similar subjects. (default: 2)")
    t9_parser.add_argument("-m", type=int, default=2,
                           help="Enter value of m for m most significant subjects. (default: 2)")
    t9_parser.add_argument(
        "--subjectid1", help="Enter first subject", type=int, default="1")
    t9_parser.add_argument(
        "--subjectid2", help="Enter second subject", type=int, default="2")
    t9_parser.add_argument(
        "--subjectid3", help="Enter third subject", type=int, default="3")
    args = arg_parser.parse_args()

    return args


def preprocess(args):
    phase1.task2(args)


def main():
    args = parse_args()
    task = args.commands
    args.image_dir = "test-image-sets/all"
    # Preprocess to generate FD for given image directory
    args_copy = copy.deepcopy(args)
    preprocess(args_copy)
    if task == "task-1":
        task1.run(args)
    if task == "task-2":
        task2.run(args)
    if task == "task-3":
        task3.run(args)
    if task == "task-4":
        task4.run(args)
    if task == "task-5":
        task5.run(args)
    if task == "task-6":
        task6.run(args)
    if task == "task-7":
        task7.run(args)
    if task == "task-8":
        task8.run(args)
    if task == "task-9":
        task9.run(args)


if __name__ == "__main__":
    main()
