#!/bin/bash

#mydir='/home/anne/Documents/featurecloud/pca/vertical-pca/data/mnist/raw'
mydir=$1
mkdir -p $mydir/raw
cd $mydir/raw
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz


mydir='/home/anne/Documents/featurecloud/pca/vertical-pca/data/mnist/splits'
mkdir -p $mydir