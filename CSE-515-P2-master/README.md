#### This Code implements Multimedia and Web Databases Phase-2 of Project. This file briefly explains how to run this program and sample outputs for each Task
</br>

## **Setup Requirements**
Python>=3.6  
pip install -r requirements.txt  
Note: Recommended to run on virtual env

## **How to RUN the program NOW!!?**
`python3 main.py --help` will guide how to use this tool

*TASK COMMANDS:*
```
> python3 main.py -h        
usage: main.py [-h]
               {task-1,task-2,task-3,task-4,task-5,task-6,task-7,task-8,task-9}
               ...

positional arguments:
  {task-1,task-2,task-3,task-4,task-5,task-6,task-7,task-8,task-9}
                        sub-command help
    task-1              First task options
    task-2              Second task options
    task-3              Third task options
    task-4              Fourth task options
    task-5              Fifth task options
    task-6              Sixth task options
    task-7              SEVENTH task options
    task-8              Eight task options
    task-9              Ninth task options

optional arguments:
  -h, --help            show this help message and exit
```
*TASK-1 SUB-COMMANDS:*
```
> python3 main.py task-1 -h
usage: main.py task-1 [-h]
                      [-x {cc,con,detail,emboss,jitter,neg,noise01,noise02,original,poster,rot,smooth,stipple}]
                      [-k K] [--fd-model {cm,elbp,hog}]
                      [--dim-reduce {pca,svd,lda,kmeans}]
                      [--cm-type {mean,std,skew}]

optional arguments:
  -h, --help            show this help message and exit
  -x {cc,con,detail,emboss,jitter,neg,noise01,noise02,original,poster,rot,smooth,stipple}
                        Enter image type. (deafult: rot)
  -k K                  Enter value k for top k latent semantics. (deafult: 2)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices.
                        (default: cm)
  --dim-reduce {pca,svd,lda,kmeans}
                        Enter dimensionality reduction technique from
                        available choices. (default: pca)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices.
                        (default: mean)
```
*TASK-2 SUB-COMMANDS:*
```
> python3 main.py task-2 -h
usage: main.py task-2 [-h] [-y [1-40]] [-k K] [--fd-model {cm,elbp,hog}]
                      [--dim-reduce {pca,svd,lda,kmeans}]
                      [--cm-type {mean,std,skew}]

optional arguments:
  -h, --help            show this help message and exit
  -y [1-40]             Enter subject. (deafult: 1)
  -k K                  Enter value k for top k latent semantics. (deafult: 2)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices.
                        (default: cm)
  --dim-reduce {pca,svd,lda,kmeans}
                        Enter dimensionality reduction technique from
                        available choices. (default: pca)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices.
                        (default: mean)
```
*TASK-3 SUB-COMMANDS:*
```
> python3 main.py task-3 -h
usage: main.py task-3 [-h] [-k K] [--fd-model {cm,elbp,hog}]
                      [--dim-reduce {pca,svd,lda,kmeans}]
                      [--cm-type {mean,std,skew}]

optional arguments:
  -h, --help            show this help message and exit
  -k K                  Enter value k for top k latent semantics. (deafult: 2)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices.
                        (default: cm)
  --dim-reduce {pca,svd,lda,kmeans}
                        Enter dimensionality reduction technique from
                        available choices. (default: pca)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices.
                        (default: mean)
```
*TASK-4 SUB-COMMANDS:*
```
> python3 main.py task-4 -h
usage: main.py task-4 [-h] [-k K] [--fd-model {cm,elbp,hog}]
                      [--dim-reduce {pca,svd,lda,kmeans}]
                      [--cm-type {mean,std,skew}]

optional arguments:
  -h, --help            show this help message and exit
  -k K                  Enter value k for top k latent semantics. (deafult: 2)
  --fd-model {cm,elbp,hog}
                        Enter feature descriptor model from available choices.
                        (default: cm)
  --dim-reduce {pca,svd,lda,kmeans}
                        Enter dimensionality reduction technique from
                        available choices. (default: pca)
  --cm-type {mean,std,skew}
                        Enter feature descriptor model from available choices.
                        (default: mean)
```
*TASK-5 SUB-COMMANDS:*
```
> python3 main.py task-5 -h
usage: main.py task-5 [-h] [--query-image-file QUERY_IMAGE_FILE]
                      [--latent-semantic-file LATENT_SEMANTIC_FILE] [-n N]

optional arguments:
  -h, --help            show this help message and exit
  --query-image-file QUERY_IMAGE_FILE
                        Enter image/s directory path
  --latent-semantic-file LATENT_SEMANTIC_FILE
                        Enter image/s directory path
  -n N                  Enter value n for top n similar images. (deafult: 3)
```
*TASK-6 SUB-COMMANDS:*
```
> python3 main.py task-6 -h
usage: main.py task-6 [-h] [--query-image-file QUERY_IMAGE_FILE]
                      [--latent-semantic-file LATENT_SEMANTIC_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --query-image-file QUERY_IMAGE_FILE
                        Enter image/s directory path
  --latent-semantic-file LATENT_SEMANTIC_FILE
                        Enter image/s directory path
```
*TASK-7 SUB-COMMANDS:*
```
> python3 main.py task-7 -h
usage: main.py task-7 [-h] [--query-image-file QUERY_IMAGE_FILE]
                      [--latent-semantic-file LATENT_SEMANTIC_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --query-image-file QUERY_IMAGE_FILE
                        Enter image/s directory path
  --latent-semantic-file LATENT_SEMANTIC_FILE
                        Enter image/s directory path
```
*TASK-8 SUB-COMMANDS:*
```
> python3 main.py task-8 -h
usage: main.py task-8 [-h] [--sss_matrix_path SSS_MATRIX_PATH]
                      [--original_images_path ORIGINAL_IMAGES_PATH] [-n N]
                      [-m M]

optional arguments:
  -h, --help            show this help message and exit
  --sss_matrix_path SSS_MATRIX_PATH
                        Enter subject subject similarity matrix file path
  --original_images_path ORIGINAL_IMAGES_PATH
                        Enter path where original input images are present
  -n N                  Enter value n, for computing n most similar objects
                        for subject
  -m M                  Enter value m, for finding m most significant subjects
```

## *All 1-9 Tasks output file structure:*
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
├── task-6-output
│   ├── task1-PCA-hog-rot-top-40-subject-weights-latent-semantics.txt
│   ├── task2-PCA-hog-subject-1-top-40-type-weights-latent-semantics.txt
│   ├── task3-PCA-hog-top-12-type-weights-latent-semantics.txt
│   └── task4-PCA-hog-top-40-subject-weights-latent-semantics.txt
├── task-7-output
│   ├── task-1-PCA-hog-rot-top-40-subject-weights-latent-semantics.txt
│   ├── task-2-PCA-hog-subject-1-top-40-type-weights-latent-semantics.txt
│   ├── task-3-PCA-hog-top-12-type-weights-latent-semantics.txt
│   └── task-4-PCA-hog-top-40-subject-weights-latent-semantics.txt
├── task-8-output
│   ├── subject-subject-similarity-graph.txt
│   └── top_significant_subjects.png
└── task-9-output
    ├── subject-subject-similarity-graph.txt
    └── topsimilarimages.png
```

