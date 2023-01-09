require(data.table)

time.dense<-fread('/home/anne/Documents/featurecloud/singular-value-decomposition/results/movielense-dense/time.log', sep='\t')
mean(time.dense$V4)/60


time.sparse<-fread('/home/anne/Documents/featurecloud/singular-value-decomposition/results/movielense-sparse/time.log', sep='\t')
mean(time.sparse$V4)/60

time.sparse<-fread('/home/anne/Documents/featurecloud/singular-value-decomposition/results/movielense-dense-8-cores/time.log', sep='\t')
mean(time.sparse$V4)/60

