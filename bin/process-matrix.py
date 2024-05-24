#!/usr/bin/env python
# coding: utf-8
"""Expression counts matrix construction."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
# from sklearn.decomposition import PCA
# import umap

# from .expression_matrix import ExpressionMatrix  # noqa: ABS101
# from .util import get_named_logger, wf_parser  # noqa: ABS101


def write_mm(df, features, barcodes, prefix=''):
    """
    df should be Pandas dataframe with columns 'feature', 'barcode', 'count'
    features should be a list of *all* features that *could* have counts
    barcodes should be a list of *all* barcodes that *could* have counts
    """

    matrix_file = f'{prefix}matrix.mtx'
    feature_file = f'{prefix}features.tsv'
    barcode_file = f'{prefix}barcodes.tsv'

    feature_to_idx = {v: k for k, v in dict(enumerate(features, 1)).items()}
    barcode_to_idx = {v: k for k, v in dict(enumerate(barcodes, 1)).items()}

    new_df = df.loc[:,['feature', 'barcode', 'count']]
    new_df['feature_idx'] = new_df.feature.map(feature_to_idx)
    new_df['barcode_idx'] = new_df.barcode.map(barcode_to_idx)

    with open(matrix_file, 'w') as f:
        if np.issubdtype(new_df['count'].values[0], np.integer):
            f.write('%%MatrixMarket matrix coordinate integer general\n')
        else:
            f.write('%%MatrixMarket matrix coordinate real general\n')
        f.write('%\n')
        f.write('{} {} {}\n'.format(len(features), len(barcodes), len(df)))
        new_df[['feature_idx', 'barcode_idx', 'count']].to_csv(f, sep=' ', index=False, header=False)

    pd.DataFrame({'x': features}).to_csv(feature_file, index=False, header=False)
    pd.DataFrame({'x': barcodes}).to_csv(barcode_file, index=False, header=False)

    return (matrix_file, feature_file, barcode_file)



parser = argparse.ArgumentParser()

parser.add_argument(
    "input", type=Path, nargs='+',
    help="TSV with read tag data.")
parser.add_argument(
    "--feature", default="gene", choices=["gene", "transcript"],
    help="Feature to compute matrix. Only used when read tag input is given.")
parser.add_argument(
    "--raw", default="raw_feature_bc_matrix",
    help="Output folder for raw counts MEX data.")
parser.add_argument(
    "--processed", default="processed_feature_bc_matrix",
    help="Output folder for processed counts MEX data.")
parser.add_argument(
    "--per_cell_expr", default="expression.mean-per-cell.tsv", type=Path,
    help="Output TSV for per-cell mean expression level.")
parser.add_argument(
    "--per_cell_mito", default="expression.mito-per-cell.tsv", type=Path,
    help="Output TSV for per-cell mean expression level.")
parser.add_argument(
    "--text", action="store_true", help=argparse.SUPPRESS)

grp = parser.add_argument_group("Filtering")
grp.add_argument(
    "--enable_filtering", action="store_true",
    help="Enable filtering of matrix.")
grp.add_argument(
    "--min_features", type=int, default=100,
    help="Filter out cells that contain fewer features than this.")
grp.add_argument(
    "--min_cells", type=int, default=3,
    help="Filter out features that are observed in fewer than this "
            "number of cells")
grp.add_argument(
    "--max_mito", type=int, default=5,
    help="Filter out cells where more than this percentage of counts "
            "belong to mitochondrial features.")
grp.add_argument(
    "--mito_prefixes", default=["MT-"], nargs='*',
    help="prefixes to identify mitochondrial features.")
grp.add_argument(
    "--norm_count", type=int, default=10000,
    help="Normalize to this number of counts per cell as "
            "is performed in CellRanger.")
grp.add_argument(
    "--filtered_mex", default="filtered_feature_bc_matrix",
    help="Output folder for raw counts MEX data.")

grp = parser.add_argument_group("UMAP creation")
grp.add_argument(
    "--enable_umap", action="store_true",
    help="Perform UMAP on matrix.")
grp.add_argument(
    "--umap_tsv", default="expression.umap.tsv", type=Path,
    help=(
        "UMAP TSV output file path. If --replicates is greater than 1 "
        "files will be named: name.index.tsv."))
grp.add_argument(
    "--replicates", type=int, default=1,
    help="Number of UMAP replicated to perform.")
grp.add_argument(
    "--pcn", type=int, default=100,
    help="Number of principal components to generate prior to UMAP")
grp.add_argument(
    "--dimensions", type=int, default=2,
    help="Number of dimensions in UMAP embedding")
grp.add_argument(
    "--min_dist", type=float, default=0.1,
    help="Minimum distance parameter of UMAP")
grp.add_argument(
    "--n_neighbors", type=int, default=15,
    help="Number of neighbors parameter of UMAP")

args = parser.parse_args()



FEATURE = args.feature
PER_CELL_MITO = args.per_cell_mito
mito_prefixes = args.mito_prefixes
RAW = args.raw
PROCESSED = args.processed
MIN_FEATURES = args.min_features
MIN_CELLS = args.min_cells
MAX_MITO = args.max_mito
PER_CELL_MITO = args.per_cell_mito
NORM_COUNT = args.norm_count
PER_CELL_EXPR = args.per_cell_expr

matrix = []

for f in args.input:
    tmp = pd.read_csv(f, sep='\t')
    tmp = tmp[['corrected_barcode', 'corrected_umi', FEATURE]].groupby(['corrected_barcode', FEATURE]).corrected_umi.nunique().reset_index()
    tmp = tmp[tmp[FEATURE] != '-']
    matrix.append(tmp)
matrix = pd.concat(matrix).groupby(['corrected_barcode', FEATURE]).corrected_umi.sum().reset_index()
matrix = matrix.rename(columns={'corrected_barcode': 'barcode', FEATURE: 'feature', 'corrected_umi': 'count'})

# write an MEX file
write_mm(matrix, matrix.feature.unique().tolist(), matrix.barcode.unique().tolist(), RAW)


# filter cells and features
feature_observed_in_n_cells = matrix.groupby('feature').barcode.nunique()
n_gene_observed_in_cell = matrix.groupby('barcode').feature.nunique()
keep_features = feature_observed_in_n_cells[feature_observed_in_n_cells>=MIN_CELLS].index.to_list()
keep_cells = n_gene_observed_in_cell[n_gene_observed_in_cell>=MIN_FEATURES].index.to_list()
matrix = matrix[matrix.barcode.isin(keep_cells)]
matrix = matrix[matrix.feature.isin(keep_features)]

# add is_mito column
all_features = matrix.feature.unique()
mito_features = set()
for prefix in mito_prefixes:
    for feature in all_features:
        if feature.startswith(prefix):
            mito_features.add(feature)
matrix['is_mito'] = matrix.feature.isin(mito_features).astype(int)

# compute mito_pct
mito_pct = matrix.groupby('barcode').is_mito.mean().reset_index().rename(columns={'barcode': 'CB'})
mito_pct['mito_pct'] = 100*mito_pct.is_mito
mito_pct[['CB', 'mito_pct']].to_csv(PER_CELL_MITO, sep='\t', index=False)
keep_mito = mito_pct[mito_pct.mito_pct<=MAX_MITO].CB.unique()
matrix = matrix[matrix.barcode.isin(keep_mito)]


# normalize and transform
total_per_cell = matrix.groupby('barcode')['count'].sum()
scaling = NORM_COUNT / total_per_cell
matrix['scaled'] = [scaling[barcode] * count for barcode, count in zip(matrix.barcode, matrix['count'])]

# TODO: this is transforming in a different manner as the default, so processed expression and mean_expression will not be comparable with default pipeline
matrix['transformed'] = np.log10(matrix.scaled)

# write an MEX file
write_mm(matrix[['feature', 'barcode', 'transformed']].rename(columns={'transformed': 'count'}), matrix.feature.unique().tolist(), matrix.barcode.unique().tolist(), PROCESSED)

# write CB, mean_expression
matrix.groupby('barcode').transformed.mean().reset_index().rename(columns={'transformed': 'mean_expression', 'barcode': 'CB'}).to_csv(PER_CELL_EXPR, sep='\t', index=False)


# output dummy UMAP files for now
fake_umap = matrix[['barcode']].rename(columns={'barcode': 'CB'})
fake_umap['D1'] = 0
fake_umap['D2'] = 0
fake_umap.to_csv(f'{FEATURE}.expression.umap.1.tsv', sep='\t', index=False)