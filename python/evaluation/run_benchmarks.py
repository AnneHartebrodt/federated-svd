import os
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=6


import time

import numpy as np

from svd.algorithms.randomized import *
from svd.algorithms.power_iteration import *
from svd.algorithms.approximate_horizontal import *

import argparse as ap
import data_import.mnist_import as mi
import data_import.spreadsheet_import as si
from scipy.sparse import coo_matrix
import data_import.gwas_import as gi
from svd.logging import *

from evaluation.data_aggregation import *
import scipy.sparse as sps


####### BENCHMARK RUNNER #######
def the_epic_loop(data, dataset_name, maxit, nr_repeats, k, splits, outdir, epsilon=1e-9, precomputed_pca=None,
                  unequal=False, guo_epsilon=1e-11, ortho_freq=1, algorithms = ['RANDOMIZED', 'GUO', 'AI-FULL', 'RI-FULL']):
    """
    run the simulation of a federated run of vertical power iteration
    Args:
        data: data frame or list of data frames containing dimension which is split
        in the columns
        dataset_name: Name, for logging
        maxit: maximal iterations to run
        nr_repeats: number of times to repeat experiments
        k: targeted dimensions
        splits: array of splits for dataset (only applicable when data is not list)
        outdir: result directory

    Returns:

    """
    # g = gv.standalone(data, k)
    islist = False
    if isinstance(data, list):
        islist = True
        data_list = data
        data = np.concatenate(data, axis=1)
        splits = [1]  # data is already split, only counter experiments need to be run.

    u, s, v = lsa.svds(data.T, k=k)
    u = np.flip(u, axis=1)
    s = np.flip(s)
    v = np.flip(v.T, axis=1)
    dataset_name_guo = dataset_name + '_guo'

    current_split = 0
    for c in range(nr_repeats):
        np.random.seed(c)
        for s in splits:
            # split the data
            if not islist:
                if unequal:
                    # if unequal then the number of sites is the length of s and s itself contains the splits
                    data_list, choice = sh.partition_data_vertically(data, len(s), randomize=True, perc=s, equal=False)
                    s = current_split
                    current_split += 1
                else:
                    data_list, choice = sh.partition_data_vertically(data, s, randomize=True)
            else:
                # make a dummy choice vector
                choice = range(data.shape[1])

            logftime = op.join(outdir, 'time.log')

            # # simulate the run

            start = time.monotonic()
            fedqr = False
            if 'RI-FULL' in algorithms:
                # simultaneous only H
                grad = False
                mode = 'power-iteration'
                print('power - matrix - ' + mode)
                outdir_gradient = op.join(outdir, 'matrix', str(s), mode)
                os.makedirs(outdir_gradient, exist_ok=True)
                print(outdir_gradient)
                filename = create_filename(outdir_gradient, dataset_name + '_' + mode, s, c, k, maxit, start)
                print(filename)
                simulate_subspace_iteration(data_list, k, maxit=maxit, u=u, filename=filename, choices=choice,
                                            precomputed_pca=precomputed_pca, federated_qr=fedqr, v=v, gradient=grad,
                                            epsilon=epsilon, g_ortho_freq=ortho_freq)
                end = time.monotonic()
                log_time(logftime, 'qr_scheme' + '_' + mode, end - start, s, c)

            if 'AI-FULL' in algorithms:
                # simultaneous only H with approx
                grad = False
                mode = 'power-approx'
                print('power - matrix - ' + mode)
                outdir_gradient = op.join(outdir, 'matrix', str(s), mode)
                os.makedirs(outdir_gradient, exist_ok=True)
                filename = create_filename(outdir_gradient, dataset_name + '_' + mode, s, c, k, maxit, start)

                print('Approximtate-init')
                g_init, h_init = approximate_vertical(data_list, k=k, factor_k=2)
                G_i = np.concatenate(g_init, axis=0)
                G_i = np.asarray(G_i)
                aol = AccuracyLogger()
                aol.open(filename)
                aol.log_current_accuracy(u=u, G_i=G_i, eigenvals=None, conv=None, current_iteration= 1,
                                         choices=choice, precomputed_pca=precomputed_pca, v=v, H_i=h_init)
                aol.close()
                print('Approximate-subspace')
                simulate_subspace_iteration(data_list, k, maxit=maxit, u=u, filename=filename, choices=choice,
                                            precomputed_pca=precomputed_pca, federated_qr=fedqr, v=v, gradient=grad,
                                            epsilon=epsilon, g_ortho_freq=ortho_freq, g_init=g_init)
                end = time.monotonic()
                log_time(logftime, 'qr_scheme' + '_' + mode, end - start, s, c)

            # Run Guo version
            if 'GUO' in algorithms:
                # Sequential
                grad = True
                fedqr = False
                grad_name = 'gradient'
                mode = 'gradient'
                print('gradient - sequential - '+ mode)
                outdir_gradient = op.join(outdir, 'vector', str(s), mode)
                os.makedirs(outdir_gradient, exist_ok=True)

                filename = create_filename(outdir_gradient, dataset_name_guo + '_' + mode, s, c, k, maxit, start)

                start = time.monotonic()
                compute_k_eigenvectors(data_list, k=k, maxit=maxit, u=u, filename=filename, choices=choice,
                                       precomputed_pca=precomputed_pca, federated_qr=fedqr, v=v, gradient=grad,
                                       epsilon=epsilon, guo_epsilon=guo_epsilon)
                end = time.monotonic()
                log_time(logftime, 'guo_single' + '_' + mode, end - start, s, c)

            if 'RANDOMIZED' in algorithms:
                grad=False
                # simulate randomized
                start = time.monotonic()
                mode = 'randomized'
                print('randomized')
                outdir_approx = op.join(outdir, 'matrix', str(s), mode)
                os.makedirs(outdir_approx, exist_ok=True)
                filename = create_filename(outdir_approx, dataset_name + '_' + mode, s, c, k, maxit, start)

                run_randomized(data_list, k, I=10, u=u, filename=filename, choices=choice,
                               precomputed_pca=precomputed_pca, federated_qr=fedqr, v=v, gradient=grad,
                               epsilon=epsilon, g_ortho_freq=ortho_freq, g_init=None)
                end = time.monotonic()
                log_time(logftime, mode, end - start, s, c)
                print(mode + ' ' + str(end - start))

            logf = op.join(outdir, 'log_choices.log')
            log_choices(logf, filename, choice)




def convert_h5_to_sparse_csr(filename, out_filename, chunksize=2500):
    start = 0
    total_rows = 0

    sparse_chunks_data_list = []
    chunks_index_list = []
    columns_name = None
    while True:
        df_chunk = pd.read_hdf(filename, start=start, stop=start + chunksize)
        if len(df_chunk) == 0:
            break
        chunk_data_as_sparse = sc.sparse.csr_matrix(df_chunk.to_numpy())
        sparse_chunks_data_list.append(chunk_data_as_sparse)
        chunks_index_list.append(df_chunk.index.to_numpy())

        if columns_name is None:
            columns_name = df_chunk.columns.to_numpy()
        else:
            assert np.all(columns_name == df_chunk.columns.to_numpy())

        total_rows += len(df_chunk)
        print(total_rows)
        if len(df_chunk) < chunksize:
            del df_chunk
            break
        del df_chunk
        start += chunksize

    all_data_sparse = sc.sparse.vstack(sparse_chunks_data_list)
    del sparse_chunks_data_list

    return  all_data_sparse




####### BENCHMARK RUNNER #######

if __name__ == '__main__':
    local=False
    np.random.seed(11)
    if local:
        start = time.monotonic()
        #data, sample_ids, variable_names = si.data_import('/home/anne/Documents/featurecloud/singular-value-decomposition/data/tabular/movielens.tsv', header=None, rownames=None, sep='\t')
        data, test_lables = mi.load_mnist('/home/anne/Documents/featurecloud/pca/vertical-pca/data/mnist/raw', 'train')
        #data, test_labels = mi.load_mnist(input_dir, 'train')
        data = coo_matrix.asfptype(data)
        #
        dataset_name = 'mnist'
        #data = pd.read_csv('/home/anne/Downloads/ml-100k/u.data', header=None, sep='\t')
        #data = sps.csc_matrix((data.iloc[:, 3], (data.iloc[:, 0], data.iloc[:, 1])), dtype='float32')
        #print(data.shape)
        #data = data.todense()
        #print(data.shape)
        scale = True
        center = True
        if scale or center:
            means = data.mean(axis=0)
            #data = data.todense()
            std = np.std(data, axis=0)
            if center:
                data = np.subtract(data, means)

            # self.tabdata.scaled = np.delete(self.tabdata.scaled, remove)
            # self.tabdata.rows = np.delete(self.tabdata.rows, remove)
            if scale:
                data = data / std

            if center:
                # impute. After centering, the mean should be 0, so this effectively mean imputation
                data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

        #dataset_name = 'movielens'
        maxit = 500
        nr_repeats = 10
        k = 10
        splits = [5, 10]
        #outdir = '/home/anne/Documents/featurecloud/singular-value-decomposition/results/mnist'
        #algorithms = 'RANDOMIZED,GUO,AI-FULL,RI-FULL'
        algorithms = 'GUO'
        algorithms = algorithms.strip().split(',')
        tolerance = 1e-6
        guo_tolerance = 1e-8
        outdir = '/home/anne/Documents/featurecloud/singular-value-decomposition/results/mnist-6'
        the_epic_loop(data, dataset_name, maxit, nr_repeats, k, splits, outdir, epsilon=tolerance, guo_epsilon=guo_tolerance,
                                           unequal=False, precomputed_pca=None, ortho_freq=1000, algorithms=algorithms)
        print('TIME: '+ str(time.monotonic() - start))
        outd = ['matrix', 'vector']
        for od in outd:
            basepath = op.join(outdir,od)
            for suf in ['.angles.u', '.angles.v', '.mev.u', 'mev.v', '.transmission', '.eo' ]:
                try:
                    create_dataframe(basepath=basepath, suffix=suf)
                except:
                    pass

    else:
        parser = ap.ArgumentParser(description='Split datasets and run "federated PCA"')
        parser.add_argument('-f', metavar='file', type=str, help='filename of data file; default tab separated')
        parser.add_argument('--filetype', metavar='filetype', type=str, help='Type of the dataset')
        parser.add_argument('--sep', metavar='sep', type=str, help='spreadsheet separator, default tab', default='\t')
        parser.add_argument('--variance', action='store_true', help='spreadsheet separator, default tab')
        parser.add_argument('--center', action='store_true', help='center data')
        parser.add_argument('-o', metavar='outfile', type=str, help='output directory')
        parser.add_argument('-r', metavar='repeats', type=int, default=20, help='Number of times to repeat experiment')
        parser.add_argument('-k', metavar='dim', default=10, type=int, help='Number of PCs to calculate')
        parser.add_argument('-t', metavar='tolerance', default=1e-9, type=float, help='Convergence tolerance')
        parser.add_argument('-s', metavar='sites', default='2,3,5,10', type=str,
                            help='comma separated list of number of sites to simulate, parsed as string')
        parser.add_argument('-i', metavar='iteration', default=2000, type=int, help='Maximum number of iterations')
        parser.add_argument('--header', metavar='iteration', default=None, type=int, help='header lines')
        parser.add_argument('--rownames', metavar='iteration', default=None, type=int, help='rownames')
        parser.add_argument('--names', metavar='iteration', default=None, type=str, help='names')
        parser.add_argument('--compare_pca', metavar='compare', default=None, type=str,
                            help='filename of precomputed pca to be compared to')
        parser.add_argument('--orthovector', metavar='compare', default=None, type=str,
                            help='filename of orthogonal file')
        parser.add_argument('--scaled', action='store_true', help='data is prescaled')
        parser.add_argument('--sf', type=str, help='scaled file', default=None)
        parser.add_argument('--unequal', default=None, type=str, help='split unequal, load split file')
        parser.add_argument('--ortho_freq',type=int, default=1, help='orthonormalisatio frequency for G')
        parser.add_argument('--algorithms', type=str, default='RANDOMIZED,GUO,AI-FULL,RI-FULL')
        args = parser.parse_args()

        np.random.seed(95)
        # import scaled SNP file
        path = args.f
        filetype = args.filetype
        sep = args.sep
        k = args.k

        if args.names is None:
            dataset_name = os.path.basename(args.f)
        else:
            dataset_name = args.names

        if args.unequal is None:
            s = args.s
            splits = s.strip().split(',')
            splits = np.int8(splits)
            unequal = False
        else:
            unequal = True
            split_data = pd.read_csv(args.unequal, sep='\t', header=None)
            splits = []
            for i in range(split_data.shape[0]):
                l = split_data.iloc[i, :].tolist()
                cleanedList = [x for x in l if not np.isnan(x)]
                splits.append(cleanedList)

        algorithms = args.algorithms.strip().split(',')
        print(algorithms)

        maxit = args.i
        nr_repeats = args.r
        outdir = args.o
        scale = args.variance
        center = args.center
        ortho_freq=args.ortho_freq

        print(outdir)
        nr_samples = 0
        nr_features = 0
        if filetype == 'delim-list':
            data_list = []
            for f in path.split(','):
                data, sample_ids, variable_names = si.data_import(f, sep=sep)
                if scale or center:
                    data = si.scale_center_data_columnwise(data, center=center, scale_variance=scale)
                nr_samples += data.shape[0]
                nr_features += data.shape[1]
                data_list.append(data)
            data = data_list


        elif filetype == 'delim':
            data, sample_ids, variable_names = si.data_import(path, sep=sep, header=args.header, rownames=args.rownames)
            if scale or center:
                data = si.scale_center_data_columnwise(data, center=center, scale_variance=scale)
                nr_samples = data.shape[0]
                nr_features = data.shape[1]

        elif filetype == 'mnist':
            data, test_lables = mi.load_mnist(path, 'train')
            data = coo_matrix.asfptype(data)

            if scale or center:
                data = si.scale_center_data_columnwise(data, center=center, scale_variance=scale)
                nr_samples = data.shape[0]
                nr_features = data.shape[1]
            data = data.T

        elif filetype == 'sparse':
            if path.endswith('.h5'):
                data = convert_h5_to_sparse_csr(os.path.join(path, "train_multi_targets.h5"))
            else:
                data = pd.read_csv(path, header=args.header, sep=sep, index_col=args.rownames)
            print(data.head())
            m = np.max(data.iloc[:,0])
            n = np.max(data.iloc[:,1])
            print(m)
            print(data.iloc[:,1])
            data = sps.csc_matrix((data.iloc[:, 2], (data.iloc[:, 0], data.iloc[:, 1])), dtype='float32')
            #remove completely empty rows and columns
            data = data[data.getnnz(1)>0]
            data = data[:,data.getnnz(0)>0]
            print(data.shape)
            if scale or center:
                now= time.monotonic()
                print('Start scaling'+ str(now))
                means = data.mean(axis=0)
                data= data.todense()
                std = np.std(data, axis=0)
                if center:
                    data = np.subtract(data, means)

                # self.tabdata.scaled = np.delete(self.tabdata.scaled, remove)
                # self.tabdata.rows = np.delete(self.tabdata.rows, remove)
                if scale:
                    data = data / std

                if center:
                    # impute. After centering, the mean should be 0, so this effectively mean imputation
                    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
                print('Finish scaling: Duration '+ str(time.monotonic()-now))
            data = data.T
            print(data.shape)

        elif filetype == 'gwas':
            bim = path + '.bim'
            traw = path + '.traw'

            if not args.scaled:
                traw_nosex = gi.remove_non_autosomes(bim, traw)

                data = gi.read_scale_write(infile=traw_nosex, outfile=path + '.traw.scaled', maf=0.01)

            else:
                print('Reading: '+path+args.sf)
                data = pd.read_table(path+args.sf, header=None, sep='\t')
                data = data.values
            nr_samples = data.shape[0]
            nr_features = data.shape[1]
        else:
            raise Exception("Filetype not supported")

        if args.compare_pca is not None:
            precomputed_pca = pd.read_table(args.compare_pca, header=0, sep='\t')
            precomputed_pca = precomputed_pca.values
        else:
            precomputed_pca = None


        os.makedirs(outdir, exist_ok=True)
        the_epic_loop(data=data, dataset_name=dataset_name, maxit=maxit, nr_repeats=nr_repeats, k=k, splits=splits,
                outdir=outdir, precomputed_pca=precomputed_pca, unequal=unequal, ortho_freq=ortho_freq, algorithms=algorithms)

        outd = ['matrix', 'vector']
        for od in outd:
            basepath = op.join(outdir,od)
            for suf in ['.angles.u', '.angles.v', '.mev.u', 'mev.v', '.transmission', '.eo' ]:
                try:
                    create_dataframe(basepath=basepath, suffix=suf)
                except:
                    pass