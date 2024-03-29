import pandas as pd
import scipy as sc
import numpy as np
import os.path as op
import os
import re
import argparse as ap



def create_dataframe(basepath, suffix):
    data_list = []
    for d in os.walk(basepath):
        print(d)
        if len(d[2])>0 and len(d[1])==0:
            levels = d[0].replace( basepath, "").split(('/'))
            fff = list(filter(lambda x: re.search(re.compile(suffix), x), d[2]))
            for f in fff:
                current_data =  pd.read_csv(op.join(d[0],f), sep='\t', header=None)
                counter = 0
                for l in levels:
                    if l != "":
                        current_data["C"+str(counter)] = l
                        counter = counter+1
                print(f)
                current_data['filename']= f
                data_list.append(current_data)

    data_list = pd.concat(data_list, axis=0)
    print(op.join(basepath, 'summary'+suffix+'.tsv'))
    data_list.to_csv(op.join(basepath ,'summary'+suffix+'.tsv'), sep='\t', header=False, index=False)
    return data_list

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Split datasets and run "federated PCA"')
    parser.add_argument('-o', metavar='outdir', type=str, help='filename of data file; default tab separated')
    args = parser.parse_args()

    outdir = args.o
    outd = ['matrix', 'vector']
    for od in outd:
        basepath = op.join(outdir, od)
        for suf in ['.angles.u', '.angles.v', '.mev.u', 'mev.v', '.transmission', '.eo']:
            try:
                create_dataframe(basepath=basepath, suffix=suf)
            except:
                print(basepath)
                print('FILE not found')
