# Federated singular value decomposition

## Setup the environment
```

mkdir singular-value-decomposition
cd singular-value-decomposition

# set a bash variable
basedir=$(pwd)
git clone https://github.com/AnneHartebrodt/federated-svd

## append the directory to the python path
export $PYTHONPATH:$basedir/svd/python
mkdir data
mkdir results
```

### Get some data
#### Step 1: Get MNIST test data

```
cd data
mkdir raw
cd raw
bash $basedir/svd/bash/data_download/mnist_download.sh
cd ..
```

#### Transform it into tabular format (and scale and center)
```
mkdir tabular
python $basedir/svd/python/data_preprocessing/mnist_to_tabular.py raw tabular/mnist.tsv
```

