import numpy as np
import pandas as pd
import argparse as ap
import pandas_plink as pp


def scale(val, pi_hat):
    '''
    Scaling of genotype matrix according to Galinksy et al.
    According to plink website fastPCA uses mean imputation for missing values. However user-group seems to accept 0 imputation as well.
    Here we use the frequency of the minor allele.
    '''
    # pi_hat should normally be greater than MAF-threshold
    # Missing values are encoded as -1
    if pi_hat > 0.0 and not np.isnan(val) and val != -1:
        try:
            val = (val - 2 * pi_hat) / np.sqrt(2 * pi_hat * (1 - pi_hat))
        except RuntimeWarning:
            print('scaling failed')
    else:
        # missing values are filled with 0
        val = 0.0
    return val




def get_nr_snps(infile):
    '''
    Read first column of dataset and returns number of items
    Args:
        infile:

    Returns:

    '''
    file = pd.read_csv(infile, usecols=[0], header=None)
    return file.shape[0]

def read_scale_write_pandas(infile, outfile, maf, zero_impute=True, mean_impute=False,
                            sep = '\t', nr_snps=100000, chunksize=1000, major_2=True):
    print('scaling')
    if zero_impute and mean_impute:
        raise Exception('Mean and zero impute cannot be true at the same time')
    l = 0
    snps_removed = 0
    data = []

    data = pd.read_csv(infile, sep=sep, header=0, index_col=[0,1,2,3,4,5])

    missing = data.shape[1] * [0]
    data = data.values
    data = data.astype(int)
    nans = np.logical_or(np.logical_or(data == 0, data == 1), data == 2)
    if major_2:
        data = np.abs(data-2)
    data[nans==False] = 0
    sumS = np.nansum(data, axis=1)
    data[nans==False] = -1
    notna = np.sum(nans, axis=1)

    pi_hat = sumS / (2 * notna)
    data = np.delete(data, np.where(pi_hat<=maf)[0], axis=0)
    pi_hat = np.delete(pi_hat, np.where(pi_hat<=maf)[0])
    print(len(pi_hat))

    data = [[scale(y, p) for y, p in zip(x, pi_hat)] for x in data]

    pd.DataFrame(data).to_csv(outfile, sep=sep, mode='a', header=False, index=False )
    return data, pi_hat

def loop_scale(data, pi_hat):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j] = scale(data[i,j], pi_hat[i])
    return data


def read_scale_write(infile, outfile, maf, zero_impute=True, mean_impute=False, sep = '\t', nr_snps=100000):
    '''
    This function reads, scales, and writes a SNP file on the fly. SNPs are in
    rows, individuals in columns.
    Args:
        infile: unscaled raw file
        outfile: scaled, downsampled file name
        maf: minor allele frequency cutoff. Lower values are filtered out
        zero_impute: Zero impute missing values (default)
        mean_impute: Mean impute missing values
        nr_snps: number of SNPs to return in the output file

    Returns: None, writes a file a side effect

    '''
    print('scaling')
    if zero_impute and mean_impute:
        raise Exception('Mean and zero impute cannot be true at the same time')
    l = 0
    phl  = []
    snps_removed = 0
    data = []
    with open(infile, 'r') as reader:
        with open(outfile, 'w') as writer:
            for line in reader:
                # read file
                line = line.strip()
                lar = line.split(sep)
                if l == 0:
                    # this is somewhat awkward, but should not infringe on performance
                    # too much
                    # initialise a counter for missing values
                    s = len(lar)
                    missing = s * [0]
                # Counter for valid values
                notna = 0
                # Counter for minor allele frequncy
                sumS = 0

                for ll in range(6,len(lar)):
                    # valid values are 0, 1, and 2 (two minor alleles)
                    if lar[ll].strip() not in ['0', '1', '2']:
                        missing[ll] = missing[ll] + 1
                        lar[ll] = np.nan  # Encode missing as np.nan

                    else:
                        try:
                            lar[ll] = int(lar[ll].strip())
                            lar[ll] = np.abs(lar[ll]-2)
                            # increase counter of maf and counter for valid values
                            sumS = sumS + lar[ll]
                            notna = notna + 1
                        except ValueError:
                            lar[ll] = np.nan

                # estimate fraction of minor allele frequency from total allele frequency
                # based on valid cells
                try:
                    pi_hat = sumS / (2 * notna)

                except:
                    print(l)
                    pi_hat = 0
                phl.append(pi_hat)
                l = l + 1
                # only use snps, that do not have an extremely low AF
                if pi_hat <= maf:
                    snps_removed = snps_removed + 1
                    continue

                if zero_impute:
                    # 0 impute as per Galinsky et al.
                    lar = [str(scale(x, pi_hat)) for x in lar[6:len(lar)]]

                if mean_impute:
                    # According to plink2 doc missing values are mean imputed
                    # Plink usergroup says it is Galinksy after all.
                    # I guess assuming Hardy Weinberg it is the same?
                    lar = [scale(x, pi_hat) if not np.isnan(x) else np.nan for x in lar[6:len(lar)]]

                data.append(lar)
                stro = '\t'.join(lar)
                writer.write(stro + '\n')
    print('SNPs removed' + str(snps_removed))
    data = np.asarray(data)
    data = data.astype(np.float)
    return data, phl



def remove_set_of_snps_in_bim(bimfile, rawfile, outfile, sep='\t'):
    # read the bim file
    nonauto = pd.read_table(bimfile, sep='\t', header=None)
    # read the raw file
    da = pd.read_table(rawfile, sep=sep)
    da = da.iloc[:, 6:da.shape[1]]

    # remove SNP variation from snp id
    snp_names = [x.split('_')[0] for x in da.columns]
    remove_snps_indices = []
    remove_snps = list(set(nonauto.iloc[:, 1]) & set(snp_names))
    for x in range(len(snp_names)):
        if snp_names[x] in remove_snps:
            remove_snps_indices.append(x)

    # remove the indices from the raw file
    da = np.asarray(da.values.T, dtype=int)
    da = np.delete(da, remove_snps_indices, axis=0)

    # write the new data to file
    pd.DataFrame(da).to_csv(outfile, sep=sep, index=False, header=False)

def check_header(file):
    # read the bim file
    with open(file) as handle:
        line = handle.readline()
        line = line.split('\t')
        if line[0] == 'CHR':
            header = 0
        else:
            header = None
    return header

def remove_non_autosomes(bimfile, rawfile, sep='\t'):
    header = check_header(bimfile)
    bim = pd.read_table(bimfile, header=header, sep=sep)

    # read the raw file
    header = check_header(rawfile)
    da = pd.read_table(rawfile, sep='\t', header=header)
    # all chromosomes not between 1-22 are removed
    autosomes_char = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
            '22', '21'])
    autosomes =set(range(1,23)).union(autosomes_char)

    print('removed ' + str(len(bim[~bim.iloc[:, 0].isin(autosomes)].index)) + ' non autosomes')

    da = da.drop(bim[~bim.iloc[:, 0].isin(autosomes)].index)
    bim = bim.drop(bim[~bim.iloc[:, 0].isin(autosomes)].index)

    # remove metadata
    da = da.iloc[:, 6:da.shape[1]]
    da = np.asarray(da)
    da = da.astype('int')

    # write the new data to file
    pd.DataFrame(da).to_csv(rawfile + '.auto', sep=sep, index=False, header=False)
    pd.DataFrame(bim).to_csv(bimfile + '.auto', sep=sep, index=False, header=False)

    return rawfile + '.auto'

def import_bed(file, two_is_major=True):
    '''
    If zero is major False, the SNP values are inverted
    This has to do with the binary encoding of plinks bed files which
    changed from Plink1 to plink 2.

    Binary encoding:
    In plink1 0 is homozygous ALT; 1 missing; 2 heterozygous ALT-REF; 3 homozygous ALT
    In plink2 0 is homozygous REF; 1 heterozygous ALT-REF; 2 homozygous ALT; 3 missing

    Args:
        file: input file


    Returns: Snp array value

    '''
    myfile = pp.read_plink1_bin(file)
    values = myfile.values.T
    # 0 become -2 becomes 2
    # 1 becomes -1 becomes 1
    # 2 becomes 0 becomes 0
    # nan stays nan
    if not two_is_major:
        values = np.abs(values-2)
    return values


def compute_GRM(infile, grmfile):
    data = pd.read_table(infile, sep='\t', header=None)
    data = data.values
    # genetic relation ship matrix
    # data is scaled already, therefore we compute patient-patient cov matrix
    # divided by the number of SNPs
    grm = data.T.dot(data) / data.shape[0]
    pd.DataFrame(grm).to_csv(grmfile, sep='\t', header=None, index=None)
    return grm


if __name__ == '__main__':

    # print('This is the GWAS scaling module library')
    #
    # parser = ap.ArgumentParser(description='Split datasets and run "federated PCA"')
    # parser.add_argument('-f', metavar='infile', type=str, help='filename of data file; default tab separated')
    # parser.add_argument('-o', metavar='outfile', type=str, help='filename of data file; default tab separated')
    # parser.add_argument('-m', metavar='MAF', type=float, help='Minor allele frequency', default=0.01)
    # parser.add_argument('-b', metavar='bimfile', type=str, help='Bimfile with postitions to be removed')
    # parser.add_argument('-g', metavar='grm', type=str, default=None)
    # args = parser.parse_args()
    #
    # bimfile = '/home/anne/Documents/featurecloud/gwas/data/hapmap/CEU/hapmap3_r1_b36_fwd.CEU.thin.bim'
    # infile = '/home/anne/Documents/featurecloud/gwas/data/hapmap/CEU/plink.traw'
    # outfile = '/home/anne/Documents/featurecloud/gwas/data/hapmap/CEU/hapmap3_r1_b36_fwd.CEU.thin.traw.auto.scaled'
    #
    #
    #
    #
    # f = remove_non_autosomes(bimfile, infile)
    # print(f)
    # # print('success')
    # print(f)
    # print(args.o)
    # print(args.m)
    # read_scale_write(f, outfile, 0.01)
    #
    # # print(get_nr_snps(ie))
    # if args.g is not None:
    #     grm = compute_GRM(args.o)
    #
    #

    import pandas as pd
    import pnumpy as pn
    file= '/home/anne/Documents/data/hapmap/hapmap_r23a.traw'
    with open(file) as handle:
        line = handle.readline()
        line = line.split('\t')
        if line[0] == 'CHR':
            header = 0
        else:
            header = None
    nr_pat = len(line)
    dat  = pd.read_table(file, header=header, sep='\t')

    import time


    #remove_non_autosomes('/home/anne/Documents/data/hapmap/hapmap_r23a.bim','/home/anne/Documents/data/hapmap/hapmap_r23a.traw')
    start = time.monotonic()
    d1,p1 = read_scale_write('/home/anne/Documents/data/hapmap/hapmap_r23a.traw','/home/anne/Documents/data/hapmap/hapmap_r23a.traw.scaled', 0.01, sep='\t')
    print(time.monotonic()-start)

    start = time.monotonic()
    #d2,p2 = read_scale_write_pandas('/home/anne/Documents/data/hapmap/hapmap_r23a.traw','/home/anne/Documents/data/hapmap/hapmap_r23a.traw.scaled.1', 0.01, sep='\t', major_2=True)
    print(time.monotonic()-start)
