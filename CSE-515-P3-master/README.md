#### This Code implements Multimedia and Web Databases Phase-3 of Project. This file briefly explains how to run this program and sample outputs for each Task
</br>

## **Setup Requirements**
Python>=3.6  
pip install -r requirements.txt  
Note: Recommended to run on virtual env

## **How to RUN the program NOW!!?**
`python3 main.py --help` will guide how to use this tool

*TASK COMMANDS:*
```
> python main.py -h            
usage: main.py [-h] {task-1,task-2,task-3,task-4,task-5,task-8} ...

positional arguments:
  {task-1,task-2,task-3,task-4,task-5,task-8}
                        sub-command help
    task-1              First task options
    task-2              Second task options
    task-3              Third task options
    task-4              Fourth task options
    task-5              Fifth task options
    task-8              Eighth task options

optional arguments:
  -h, --help            show this help message and exit
```
*TASK-1 SUB-COMMANDS:*
```
> python main.py task-1 -h
usage: main.py task-1 [-h] [-k K] [--fd-model {cm,elbp,hog}]
                      [--dim-reduce {pca,svd,lda,kmeans}] [--cm-type {mean,std,skew}]
                      [--classifier {svm,dt,ppr}] [--path1 PATH1] [--path2 PATH2]

optional arguments:
  -h, --help            show this help message and exit
  -k K                  Enter value k for top k latent semantics. (deafult: 2)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices. (default: hog)
  --dim-reduce {pca,svd,lda,kmeans}
                        Enter dimensionality reduction technique from available choices.
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices. (default: mean)
  --classifier {svm,dt,ppr}
                        Enter Classification technique from available choices. (default: svm)
  --path1 PATH1         Enter file path for training data
  --path2 PATH2         Enter file path for test data
```
*TASK-2 SUB-COMMANDS:*
```
> python main.py task-2 -h
usage: main.py task-2 [-h] [-k K] [--fd-model {cm,elbp,hog}]
                      [--dim-reduce {pca,svd,lda,kmeans}] [--cm-type {mean,std,skew}]
                      [--classifier {svm,dt,ppr}] [--path1 PATH1] [--path2 PATH2]

optional arguments:
  -h, --help            show this help message and exit
  -k K                  Enter value k for top k latent semantics. (deafult: 2)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices. (default: hog)
  --dim-reduce {pca,svd,lda,kmeans}
                        Enter dimensionality reduction technique from available choices.
                        (default: pca)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices. (default: mean)
  --classifier {svm,dt,ppr}
                        Enter Classification technique from available choices. (default: svm)
  --path1 PATH1         Enter file path for training data
  --path2 PATH2         Enter file path for test data
```
*TASK-3 SUB-COMMANDS:*
```
> python main.py task-3 -h
usage: main.py task-3 [-h] [-k K] [--fd-model {cm,elbp,hog}]
                      [--dim-reduce {pca,svd,lda,kmeans}] [--cm-type {mean,std,skew}]
                      [--classifier {svm,dt,ppr}] [--path1 PATH1] [--path2 PATH2]

optional arguments:
  -h, --help            show this help message and exit
  -k K                  Enter value k for top k latent semantics. (deafult: 2)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices. (default: hog)
  --dim-reduce {pca,svd,lda,kmeans}
                        Enter dimensionality reduction technique from available choices.
                        (default: pca)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices. (default: mean)
  --classifier {svm,dt,ppr}
                        Enter Classification technique from available choices. (default: svm)
  --path1 PATH1         Enter file path for training data
  --path2 PATH2         Enter file path for test data
```
*TASK-4 SUB-COMMANDS:*
```
> python main.py task-4 -h
usage: main.py task-4 [-h] [-l L] [-k K] [--latent-semantic-file LATENT_SEMANTIC_FILE]
                      [--image-dir-path IMAGE_DIR_PATH] [--query-image-file QUERY_IMAGE_FILE]
                      [-t T] [--fd-model {cm,elbp,hog}] [--cm-type {mean,std,skew}]

optional arguments:
  -h, --help            show this help message and exit
  -l L                  Enter value l denoting the number of hash layers (deafult: 2)
  -k K                  Enter value k denoting the number of hashes per layer (deafult: 2)
  --latent-semantic-file LATENT_SEMANTIC_FILE
                        Enter latent semantic directory path(optional)
  --image-dir-path IMAGE_DIR_PATH
                        Enter query image path
  --query-image-file QUERY_IMAGE_FILE
                        Enter query image path
  -t T                  Enter value t for top t similar. (deafult: 5)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices. (default: hog)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices. (default: mean)
```
*TASK-5 SUB-COMMANDS:*
```
> python main.py task-5 -h
usage: main.py task-5 [-h] [-b B] [--latent-semantic-file LATENT_SEMANTIC_FILE]
                      [--image-dir-path IMAGE_DIR_PATH] [--query-image-file QUERY_IMAGE_FILE]
                      [-t T] [--fd-model {cm,elbp,hog}] [--cm-type {mean,std,skew}]

optional arguments:
  -h, --help            show this help message and exit
  -b B                  Enter value b denoting the number of bits per dimensions used for
                        compressing the vector data (deafult: 2)
  --latent-semantic-file LATENT_SEMANTIC_FILE
                        Enter latent semantic directory path(optional)
  --image-dir-path IMAGE_DIR_PATH
                        Enter query image path
  --query-image-file QUERY_IMAGE_FILE
                        Enter query image path
  -t T                  Enter value t for top t similar. (deafult: 5)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices. (default: hog)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices. (default: mean)
```
*TASK-8 SUB-COMMANDS:*
```
> python main.py task-8 -h
usage: main.py task-8 [-h] [--latent-semantic-file LATENT_SEMANTIC_FILE]
                      [--image-dir-path IMAGE_DIR_PATH] [--query-image-file QUERY_IMAGE_FILE]
                      [-t T] [--fd-model {cm,elbp,hog}] [--cm-type {mean,std,skew}]
                      [--index-algorithm {va,lsh}] [-b B] [-l L] [-k K]
                      [--classifier {svm,dt,ppr}]

optional arguments:
  -h, --help            show this help message and exit
  --latent-semantic-file LATENT_SEMANTIC_FILE
                        Enter latent semantic directory path(optional)
  --image-dir-path IMAGE_DIR_PATH
                        Enter query image path
  --query-image-file QUERY_IMAGE_FILE
                        Enter query image path
  -t T                  Enter value t for top t similar. (deafult: 5)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices. (default: hog)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices. (default: mean)
  --index-algorithm {va,lsh}
                        Enter index structure algorithm from available choices. (default: va)
  -b B                  Enter value b denoting the number of bits per dimensions used for
                        compressing the vector data (deafult: 100)
  -l L                  Enter value l denoting the number of hash layers (deafult: 2)
  -k K                  Enter value k denoting the number of hashes per layer (deafult: 2)
  --classifier {svm,dt,ppr}
                        Enter Classification technique from available choices. (default: svm)
```

## *All 1-5, 8 Tasks output file structure:*
Under dataset of Image directory we create a sub-directory named "task-i-output" and store all relavant outputs of i-th task. Given below is sample directory structure.
```
test-image-sets/all
├── image-0.png
├── image-stipple-9-9.png
:
:
├── task-1-output
│   ├── KMEANS-cm-mean-rot-top-40-subject-weights-latent-semantics.pkl
│   ├── KMEANS-cm-mean-rot-top-40-subject-weights-latent-semantics.txt
│   ├── KMEANS-cm-mean-rot-top-40-subject-weights.txt
│   :
│   ├── SVD-hog-original-top-40-subject-weights-latent-semantics.pkl
│   ├── SVD-hog-original-top-40-subject-weights-latent-semantics.txt
│   └── SVD-hog-original-top-40-subject-weights.txt
├── task-2-output
│   ├── KMEANS-cm-mean-subject-1-top-40-type-weights-latent-semantics.pkl
│   ├── KMEANS-cm-mean-subject-1-top-40-type-weights-latent-semantics.txt
│   ├── KMEANS-cm-mean-subject-1-top-40-type-weights.txt
│   :
│   ├── SVD-hog-subject-1-top-40-type-weights-latent-semantics.pkl
│   ├── SVD-hog-subject-1-top-40-type-weights-latent-semantics.txt
│   └── SVD-hog-subject-1-top-40-type-weights.txt
├── task-3-output
│   ├── PCA-cm-mean-top-12-type-weights-latent-semantics.pkl
│   ├── PCA-cm-mean-top-12-type-weights-latent-semantics.txt
│   ├── PCA-cm-mean-top-12-type-weights.txt
│   ├── PCA-elbp-top-12-type-weights-latent-semantics.pkl
│   ├── PCA-elbp-top-12-type-weights-latent-semantics.txt
│   ├── PCA-elbp-top-12-type-weights.txt
│   ├── PCA-hog-top-12-type-weights-latent-semantics.pkl
│   ├── PCA-hog-top-12-type-weights-latent-semantics.txt
│   ├── PCA-hog-top-12-type-weights.txt
│   ├── cm-mean-type-type-similarity.txt
│   ├── elbp-type-type-similarity.txt
│   ├── hog-type-type-similarity.txt
│   ├── nd_cm-mean.pkl
│   ├── nd_elbp.pkl
│   ├── nd_hog.pkl
│   └── cm-mean-type-type-similarity.txt
├── task-4-output
│   ├── PCA-cm-mean-top-40-subject-weights-latent-semantics.pkl
│   ├── PCA-cm-mean-top-40-subject-weights-latent-semantics.txt
│   ├── PCA-cm-mean-top-40-subject-weights.txt
│   ├── PCA-hog-top-2-subject-weights-latent-semantics.pkl
│   ├── PCA-hog-top-2-subject-weights-latent-semantics.txt
│   ├── PCA-hog-top-2-subject-weights.txt
│   ├── PCA-hog-top-40-subject-weights-latent-semantics.pkl
│   ├── PCA-hog-top-40-subject-weights-latent-semantics.txt
│   ├── PCA-hog-top-40-subject-weights.txt
│   ├── cm-mean-subject-subject-similarity.txt
│   ├── hog-subject-subject-similarity.txt
│   ├── nd_cm-mean.pkl
│   └── nd_hog.pkl
├── task-5-output
│   ├── task-1-PCA-hog-rot-top-40-subject-weights-latent-semantics.png
│   ├── task-2-PCA-hog-subject-1-top-40-type-weights-latent-semantics.png
│   ├── task-3-PCA-hog-top-12-type-weights-latent-semantics.png
│   └── task-4-PCA-hog-top-40-subject-weights-latent-semantics.png
└── task-8-output
    ├── feedback-output-svm.txt
    ├── feedback-output-dt.txt
    ├── top_relevant_images_svm.png
    └── top_relevant_images_dt.png
```
