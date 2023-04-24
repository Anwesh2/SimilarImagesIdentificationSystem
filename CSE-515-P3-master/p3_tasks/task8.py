import os
from p3_tasks import task4, task5
from utils import classifier
from utils.classifier import DT, SVM, dt_predict, svm_predict
from utils.utils import euclidean, get_image_from_dir
from p2_tasks import phase1
import copy
import math

def display(args, top_t_images, top_t_dist, query_image_file):
    original_faces = [get_image_from_dir(os.path.join(query_image_file))]
    top_t_dist.insert(0, 0)
    for i, img in enumerate(top_t_images):
        top_t_images[i] = os.path.join(args.image_dir, img)
    top_t_images.insert(0, query_image_file)
    count = 25
    for img in top_t_images:
        original_faces.append(get_image_from_dir(
            img))
        count -= 1
        if count == 1:
            break
    phase1.show_olivetti_faces(original_faces, top_t_images, top_t_dist,
                        "top n images")()

def train_model(classifier, relevant_images_input, irrelevant_images_input, images_fd, img_label):
    X = []
    y = []
    for img in img_label:
        X.append(images_fd[img].ravel())
        y.append(img_label[img])
    for img in relevant_images_input:
        X.append(images_fd[img])
        y.append("relevant")
        img_label[img] = "relevant"
    for img in irrelevant_images_input:
        X.append(images_fd[img])
        y.append("irrelevant")
        img_label[img] = "irrelevant"
    if classifier == "svm":
        return SVM(X, y)
    else:
        return DT(X, y)

def get_input(input_str, valid_opts, default=None):
    '''
    Get input from user
    '''
    val = input(input_str)
    while True:
        if val == '' and default is not None:
            val = default
            break
        if val.strip().lower() in valid_opts:
            break
        val = input(input_str + ' ')
    ret = val.strip().lower()
    return ret


def get_boolean_input(input_str, default):
    default_yes = ' [Y/n] '
    default_no = ' [y/N] '
    input_str = input_str + default_yes if default else input_str + default_no
    res = get_input(input_str, ['y', 'n', 'yes', 'no'], 'yes' if default else 'no')
    if res == 'y' or res == 'yes':
        return True
    return False

def validate_images(input, expected):
    not_present = []
    for img in input:
        if img not in expected:
            not_present.append(img)
    if not_present:
        print("These images not found in top images returned above: ", not_present)
        print("Please enter correct images from above top images")
        return False
    return True

def run(args):
    args.image_dir = args.image_dir_path
    index_function = task4.run
    if args.index_algorithm == "va":
        index_function = task5.run
    original_query_image_file = args.query_image_file
    top_t = args.t
    step_increment = min(math.floor(len([name for name in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, name))])/3), 100)
    # index is created to find top_t images only and we later need index to find top_t + 50
    args.t = top_t + step_increment
    print("Step increment: ", step_increment)
    top_t_images, top_t_dist, images_fd, original_query_image_fd, index = index_function(args, print_flag=False)
    top_t_images = top_t_images[:top_t]
    top_t_dist = top_t_dist[:top_t]
    top_t_images_copy = copy.deepcopy(top_t_images)
    top_t_dist_copy = copy.deepcopy(top_t_dist)
    display(args, top_t_images_copy, top_t_dist_copy, original_query_image_file)
    table_data = []
    for img, dist in zip(top_t_images, top_t_dist):
        table_data.append([img, dist, "NOT-LABELED"])
    print("=====================================================")
    print("{:<25}| {:<20}| {:<20}".format("IMAGE", "DISTANCE", "LABEL"))
    print("=====================================================")
    for row in table_data:
        print("{:<25}| {:<20.3f}| {:<20}|".format(*row))
    img_label = {}
    query_images_queue = []
    used_query_image = []
    final_top_t_images = {}
    while not get_boolean_input("Are all images relevant?", False):
        relevant_images_input = []
        irrelevant_images_input = []
        while(True):
            relevant_images_input = input("\nEnter comma separated relevant images: ").split(",")
            if relevant_images_input[0] == "":
                print("Classifier needs atleast one relevant and one irrelevant image")
                continue
            if not validate_images(relevant_images_input, top_t_images):
                continue
            irrelevant_images_input = input("\nEnter comma separated irrelevant images: ").split(",")
            if irrelevant_images_input[0] == "":
                print("Classifier needs atleast one relevant and one irrelevant image")
                continue
            if not validate_images(irrelevant_images_input, top_t_images):
                continue
            common_images = set(relevant_images_input) & set(irrelevant_images_input)
            if common_images:
                print("{} images cannot be both relevant and irrelevant! Please try again".format(common_images))
                continue
            break
        clf = train_model(args.classifier, relevant_images_input, irrelevant_images_input, images_fd, img_label)
        for img in relevant_images_input:
            final_top_t_images[img] = 0
            if img not in used_query_image:
                query_images_queue.append(img)
        
        if not query_images_queue:
            query_images_queue = used_query_image.pop(0)
            used_query_image = []
        args.query_image_file = os.path.join(args.image_dir_path, query_images_queue.pop(0))
        args.t = top_t + step_increment
        print("Query image considered: ", args.query_image_file)
        print("Deep copying index")
        index_copy = copy.deepcopy(index)
        print("Deep copy done! Searching index structure")
        top_t_images, top_t_dist, images_fd, query_image_fd, _ = index_function(args, print_flag=False, index=index_copy)
        if args.classifier == "svm":
            predict_fun = svm_predict
        else:
            predict_fun = dt_predict
        new_top_t_images = []
        new_top_t_dist = []
        top_t_img_dist = {}
        discarded = 0

        for img in top_t_images:
            pred_label = predict_fun(clf, [images_fd[img]])
            if pred_label == "relevant":
                new_top_t_images.append(img)
            else:
                if img in final_top_t_images:
                    final_top_t_images.pop(img, None)
                # print("irrelevant: ", img)
                discarded += 1
        print("Images discarded using feedback: ", discarded, " out of nearest ", args.t)
        new_top_t_images = list(set(new_top_t_images) - set(final_top_t_images.keys()))
        for img in new_top_t_images:
            dist = euclidean(images_fd[img], original_query_image_fd)
            new_top_t_dist.append(dist)
            top_t_img_dist[img] = dist
        
        sorted_top_t_img_dist = dict(
        sorted(top_t_img_dist.items(), key=lambda item: item[1]))
        table_data = []
        count = top_t - len(final_top_t_images)
        unlabelled_top_images = []
        for img in sorted_top_t_img_dist:
            unlabelled_top_images.append(img)
            count -=1
            if count == 0:
                break
        new_top_t_images = list(set(unlabelled_top_images).union(set(final_top_t_images.keys())))
        new_top_t_dist = []
        top_t_img_dist = {}
        for img in new_top_t_images:
            dist = euclidean(images_fd[img], original_query_image_fd)
            new_top_t_dist.append(dist)
            top_t_img_dist[img] = dist
        final_sorted_top_t = dict(
        sorted(top_t_img_dist.items(), key=lambda item: item[1]))

        for img in final_sorted_top_t:
            if img in img_label:
                label = img_label[img]
            else:
                label = "NOT-LABELED"
            table_data.append([img, final_sorted_top_t[img], label])
        print("=====================================================")
        print("{:<25}| {:<20}| {:<20}".format("IMAGE", "DISTANCE", "LABEL"))
        print("=====================================================")
        for row in table_data:
            print("{:<25}| {:<20.3f}| {:<20}|".format(*row))
        top_t_dist = new_top_t_dist
        top_t_images = new_top_t_images
        top_t_images_copy = copy.deepcopy(top_t_images)
        top_t_dist_copy = copy.deepcopy(top_t_dist)
        display(args, top_t_images_copy, top_t_dist_copy, original_query_image_file)


