import click
from scipy.sparse._coo import coo_matrix
import pandas as pd

import python.data_import.mnist_import as mi
import python.data_import.spreadsheet_import as si

@click.command()
@click.argument('input')
@click.argument('tabular')
def transform(input, tabular):

    """Transform the byte data into tabular text data \n
    [INPUT] Directory where byte data resides \n
    [TABULAR] File to write tabular data to \n
    """
    click.echo(f"Hello {input}!")
    click.echo(f"Hello {tabular}!")

    data, test_lables = mi.load_mnist(input, 'train')
    # data, test_labels = mi.load_mnist(input_dir, 'train')
    data = coo_matrix.asfptype(data)
    data = si.scale_center_data_columnwise(data, center=True, scale_variance=False)
    pd.DataFrame(data).to_csv(tabular, header=True, index=True, sep='\t')



if __name__ == '__main__':
    transform()