
from p2_tasks import phase1
from p3_tasks import task1, task2, task3, task4, task5, task8
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
    # t1_parser.add_argument("-y", type=int,
    #                        help="Enter subject", choices=range(1, 41), metavar="[1-40]")
    t1_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k for top k latent semantics. (deafult: 2)")
    t1_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'], help="Enter feature descriptor model from available choices. (default: hog)", type=str, default="hog")
    t1_parser.add_argument(
        "--dim-reduce", choices=['pca', 'svd', 'lda', 'kmeans'], help="Enter dimensionality reduction technique from available choices.", type=str)
    t1_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"], help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    t1_parser.add_argument("--classifier", choices=["svm", "dt", "ppr"], help="Enter Classification technique from available choices. (default: svm)", type=str, default="svm")
    t1_parser.add_argument("--path1", type=str, default="test-image-sets/500/",
                           help="Enter file path for training data")
    t1_parser.add_argument("--path2", type=str, default="test-image-sets/100",
                           help="Enter file path for test data")
    
    t2_parser = subparsers.add_parser("task-2", help="Second task options")
    # t2_parser.add_argument("-x", type=str, choices=["cc", "con", "detail", "emboss", "jitter", "neg", "noise01", "noise02",
    #                        "original", "poster", "rot", "smooth", "stipple"], default="rot", help="Enter image type. (deafult: rot)")
    t2_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k for top k latent semantics. (deafult: 2)")
    t2_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'], help="Enter feature descriptor model from available choices. (default: hog)", type=str, default="hog")
    t2_parser.add_argument(
        "--dim-reduce", choices=['pca', 'svd', 'lda', 'kmeans'], help="Enter dimensionality reduction technique from available choices. (default: pca)", type=str)
    t2_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"], help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    t2_parser.add_argument("--classifier", choices=["svm", "dt", "ppr"], help="Enter Classification technique from available choices. (default: svm)", type=str, default="svm")
    t2_parser.add_argument("--path1", type=str, default="test-image-sets/500/",
                           help="Enter file path for training data")
    t2_parser.add_argument("--path2", type=str, default="test-image-sets/100",
                           help="Enter file path for test data")
    
    t3_parser = subparsers.add_parser("task-3", help="Third task options")
    # t3_parser.add_argument("-x", type=str, choices=["cc", "con", "detail", "emboss", "jitter", "neg", "noise01", "noise02",
    #                        "original", "poster", "rot", "smooth", "stipple"], default="rot", help="Enter image type. (deafult: rot)")
    t3_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k for top k latent semantics. (deafult: 2)")
    t3_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'], help="Enter feature descriptor model from available choices. (default: hog)", type=str, default="hog")
    t3_parser.add_argument(
        "--dim-reduce", choices=['pca', 'svd', 'lda', 'kmeans'], help="Enter dimensionality reduction technique from available choices. (default: pca)", type=str)
    t3_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"], help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    t3_parser.add_argument("--classifier", choices=["svm", "dt", "ppr"], help="Enter Classification technique from available choices. (default: svm)", type=str, default="svm")
    t3_parser.add_argument("--path1", type=str, default="test-image-sets/500/",
                           help="Enter file path for training data")
    t3_parser.add_argument("--path2", type=str, default="test-image-sets/100",
                           help="Enter file path for test data")
    
    t4_parser = subparsers.add_parser("task-4", help="Fourth task options")
    t4_parser.add_argument("-l", type=int, default=2,
                           help="Enter value l denoting the number of hash layers (deafult: 2)")
    t4_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k denoting the number of hashes per layer (deafult: 2)")
    t4_parser.add_argument("--latent-semantic-file", type=str, help="Enter latent semantic directory path(optional)")
    t4_parser.add_argument("--image-dir-path", type=str,
                           default="test-image-sets/500", help="Enter query image path")
    t4_parser.add_argument("--query-image-file", type=str,
                           default="test-image-sets/500/image-original-1-5.png", help="Enter query image path")
    t4_parser.add_argument("-t", type=int, default=5,
                           help="Enter value t for top t similar. (deafult: 5)")
    t4_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'],
        help="Enter feature descriptor model from available choices. (default: hog)", type=str, default="hog")
    t4_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"],
        help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    
    t5_parser = subparsers.add_parser("task-5", help="Fifth task options")
    t5_parser.add_argument("-b", type=int, default=2,
                           help="Enter value b denoting the number of bits per dimensions used for compressing the vector data (deafult: 2)")
    t5_parser.add_argument("--latent-semantic-file", type=str, help="Enter latent semantic directory path(optional)")
    t5_parser.add_argument("--image-dir-path", type=str,
                           default="test-image-sets/500", help="Enter query image path")
    t5_parser.add_argument("--query-image-file", type=str,
                           default="test-image-sets/500/image-original-1-5.png", help="Enter query image path")
    t5_parser.add_argument("-t", type=int, default=5,
                           help="Enter value t for top t similar. (deafult: 5)")
    t5_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'],
        help="Enter feature descriptor model from available choices. (default: hog)", type=str, default="hog")
    t5_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"],
        help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    
    t8_parser = subparsers.add_parser("task-8", help="Eighth task options")
    t8_parser.add_argument("--latent-semantic-file", type=str, help="Enter latent semantic directory path(optional)")
    t8_parser.add_argument("--image-dir-path", type=str,
                           default="test-image-sets/500", help="Enter query image path")
    t8_parser.add_argument("--query-image-file", type=str,
                           default="test-image-sets/500/image-original-1-5.png", help="Enter query image path")
    t8_parser.add_argument("-t", type=int, default=5,
                           help="Enter value t for top t similar. (deafult: 5)")
    t8_parser.add_argument(
        "--fd-model", choices=['cm', 'elbp', 'hog'],
        help="Enter feature descriptor model from available choices. (default: hog)", type=str, default="hog")
    t8_parser.add_argument(
        "--cm-type", choices=["mean", "std", "skew"],
        help="Enter feature descriptor model from available choices. (default: mean)", default="mean")
    t8_parser.add_argument("--index-algorithm", choices=['va', 'lsh'], help="Enter index structure algorithm from available choices. (default: va)", default="va")
    t8_parser.add_argument("-b", type=int, default=100,
                           help="Enter value b denoting the number of bits per dimensions used for compressing the vector data (deafult: 100)")
    t8_parser.add_argument("-l", type=int, default=2,
                           help="Enter value l denoting the number of hash layers (deafult: 2)")
    t8_parser.add_argument("-k", type=int, default=2,
                           help="Enter value k denoting the number of hashes per layer (deafult: 2)")
    t8_parser.add_argument("--classifier", choices=["svm", "dt", "ppr"], help="Enter Classification technique from available choices. (default: svm)", type=str, default="svm")
    args = arg_parser.parse_args()

    return args


def preprocess(args):
    phase1.task2(args)


def main():
    args = parse_args()
    task = args.commands
    # args.image_dir = "test-image-sets/all"
    # Preprocess to generate FD for given image directory
    args_copy = copy.deepcopy(args)
    # preprocess(args_copy)
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
    if task == "task-8":
        task8.run(args)

if __name__ == "__main__":
    main()
