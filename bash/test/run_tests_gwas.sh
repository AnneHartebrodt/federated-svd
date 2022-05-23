gwaspath=$1
export PYTHONPATH=$PYTHONPATH:$gwaspath/federated-svd/python
echo $PYTHONPATH
#conda activate federated-pca
datapath=$gwaspath/data/1000g/raw
resultpath=$gwaspath/results/1000g

mkdir -p $resultpath
# take 2 chromosomes, we don't want to spam
for e in {1..2} ;
do
for i in 100000 500000 all ;
do

mkdir -p $resultpath/chr${e}

python3 $gwaspath/federated-svd/python/evaluation/run_benchmarks.py -f \
$datapath/chr${e}/chr${e} \
--filetype 'gwas' --center -o $resultpath/chr${e}.$i -r 10 -k 10 \
 -i 1000 --sep '\t' --header 0 --rownames 0 --names chr${e}.$i \
 --vert -s 5 --ortho_freq 1000 --scaled --sf '.traw.scaled.'$i
done
done

