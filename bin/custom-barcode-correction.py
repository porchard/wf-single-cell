#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import rapidfuzz
import numpy as np
import time
from scipy.stats import binom
import sys
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--whitelist', required=True)
parser.add_argument('--tag-file', required=True)
parser.add_argument('--umi-counts', required=True)
parser.add_argument('--method', required=True)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')

#WHITELIST = '/scratch/scjp_root/scjp0/porchard/2024-03-ONT/work/test-barcode-correction-no-shortlist/data/barcode_longlist_dir/737K-arc-v1.txt.gz'
#TAG_FILE = '/scratch/scjp_root/scjp0/porchard/2024-03-ONT/work/test-barcode-correction-no-shortlist/data/barcodes/10_barcode.tsv'
#UMI_COUNTS = '/scratch/scjp_root/scjp0/porchard/2024-03-ONT/work/new-barcode-correction/results/umi-counts/umi-counts.txt'
WHITELIST = args.whitelist
TAG_FILE = args.tag_file
UMI_COUNTS = args.umi_counts
METHOD = args.method

METHOD_CHOICES = ['phred', 'error_rate', 'edit_distance']
if not METHOD in METHOD_CHOICES:
    raise ValueError('--method must be one of: '.format(', '.join(METHOD_CHOICES)))


umi_counts = pd.read_csv(UMI_COUNTS, sep='\t', header=0)
umi_counts.columns = ['CR', 'umis']
umi_counts = umi_counts.set_index('CR').umis.to_dict()

whitelist = pd.read_csv(WHITELIST, sep='\t', header=None)[0].to_list()
whitelist_set = set(whitelist)

tags = pd.read_csv(TAG_FILE, sep='\t')

whitelist_match = tags[tags.CR.isin(whitelist_set)]
no_whitelist_match = tags[~tags.CR.isin(whitelist_set)]


def fast_correct(cr_list, whitelist, max_ed=2, min_ed_diff=2):
    """Use only edit distances"""
    
    if len(cr_list) != len(set(cr_list)):
        raise ValueError('cr_list cannot contain duplicates!')

    score_cutoff = max_ed + min_ed_diff

    distances = rapidfuzz.process.cdist(cr_list, whitelist, scorer=rapidfuzz.distance.Levenshtein.distance, score_cutoff=score_cutoff)
    x = pd.DataFrame(np.argwhere(distances<=score_cutoff), columns=['CR_index', 'WL_index'])
    x['ED'] = [distances[cr_index,wl_index] for cr_index, wl_index in zip(x.CR_index, x.WL_index)]
    x['CR'] = np.array(cr_list)[x.CR_index.to_list()]
    x['WL_match'] = np.array(whitelist)[x.WL_index.to_list()]
    x = x.sort_values('ED')
    
    matches = {} # CR --> CB

    for CR, df in x.groupby('CR'):
        best_edit_distance = df.ED.values[0]
        best_match = df.WL_match.values[0]
        if best_edit_distance > max_ed:
            continue
        elif best_edit_distance <= max_ed and len(df) == 1:
            matches[CR] = best_match
            continue
        elif best_edit_distance <= max_ed:
            second_best_edit_distance = df.ED.values[1]
            if second_best_edit_distance - best_edit_distance >= min_ed_diff:
                matches[CR] = best_match            
        else:
            # shouldn't be here
            raise ValueError(f'Failed to infer for CR = {CR}')

    
    for i in cr_list:
        if i not in matches:
            matches[i] = '-'
    
    return matches


def prob_of_error(uncorrected, potential_correction, phred):
    # find the error
    total_p = 1
    for i in range(len(uncorrected)):
        if uncorrected[i] != potential_correction[i]:
            q = ord(phred[i]) - 33
            p = 10**(-q/10)
            total_p *= p
    return total_p if total_p != 1 else None


def fast_correct_use_error_rate(cr_list, whitelist, umis_per_cb, max_ed_match=2, score_cutoff=3, error_rate=0.05, prob_threshold=0.95):
    cb_length = len(whitelist[0])
    for i in whitelist:
        if len(i) != cb_length:
            raise ValueError('Whitelist contains CBs of different lengths')
    if len(cr_list) != len(set(cr_list)):
        raise ValueError('cr_list cannot contain duplicates!')
            
    distances = rapidfuzz.process.cdist(cr_list, whitelist, scorer=rapidfuzz.distance.Levenshtein.distance, score_cutoff=score_cutoff)
    x = pd.DataFrame(np.argwhere(distances<=score_cutoff), columns=['CR_index', 'WL_index'])
    x['ED'] = [distances[cr_index,wl_index] for cr_index, wl_index in zip(x.CR_index, x.WL_index)]
    x['CR'] = np.array(cr_list)[x.CR_index.to_list()]
    x['WL_match'] = np.array(whitelist)[x.WL_index.to_list()]
    x = x.sort_values('ED')
    
    x['umis'] = x.WL_match.map(lambda y: 1+umis_per_cb[y] if y in umis_per_cb else 1)
    x['error_prob'] = binom.pmf(x.ED, cb_length, error_rate)
    x['prior'] = x.groupby('CR').umis.transform(lambda x: x/sum(x))
    x['prob_unnormalized'] = x.error_prob * x.prior
    x['prob_normalized'] = x.groupby('CR').prob_unnormalized.transform(lambda x: x/sum(x))
    
    x = x.sort_values('prob_normalized', ascending=False).groupby('CR').head(1)
    x = x[x.prob_normalized>=prob_threshold]
    x = x[x.ED<=max_ed_match]
    
    matches = {CR: CB for CR, CB in zip(x.CR, x.WL_match)}
    
    for i in cr_list:
        if i not in matches:
            matches[i] = '-'
    
    return matches


def fast_correct_use_phred(cr_list, phred_list, whitelist, umis_per_cb, max_ed_match=2, score_cutoff=3, error_rate=0.05, prob_threshold=0.95):
    cb_length = len(whitelist[0])
    for i in whitelist:
        if len(i) != cb_length:
            raise ValueError('Whitelist contains CBs of different lengths')
            
    distances = rapidfuzz.process.cdist(cr_list, whitelist, scorer=rapidfuzz.distance.Levenshtein.distance, score_cutoff=score_cutoff)
    x = pd.DataFrame(np.argwhere(distances<=score_cutoff), columns=['CR_index', 'WL_index'])
    x['ED'] = [distances[cr_index,wl_index] for cr_index, wl_index in zip(x.CR_index, x.WL_index)]
    x['CR'] = np.array(cr_list)[x.CR_index.to_list()]
    x['CY'] = np.array(phred_list)[x.CR_index.to_list()]
    x['WL_match'] = np.array(whitelist)[x.WL_index.to_list()]
    x = x.sort_values('ED')
    
    x['umis'] = x.WL_match.map(lambda y: 1+umis_per_cb[y] if y in umis_per_cb else 1)
    x['error_prob'] = [prob_of_error(CR, WL_match, CY) if len(CR) == cb_length else binom.pmf(ED, cb_length, error_rate) for CR, CY, WL_match, ED in zip(x.CR, x.CY, x.WL_match, x.ED)]
    x['prior'] = x.groupby(['CR', 'CY']).umis.transform(lambda x: x/sum(x))
    x['prob_unnormalized'] = x.error_prob * x.prior
    x['prob_normalized'] = x.groupby(['CR', 'CY']).prob_unnormalized.transform(lambda x: x/sum(x))
    
    x = x.sort_values('prob_normalized', ascending=False).groupby(['CR', 'CY']).head(1)
    x = x[x.prob_normalized>=prob_threshold]
    x = x[x.ED<=max_ed_match]
    
    matches = {(CR, CY): CB for CR, CY, CB in zip(x.CR, x.CY, x.WL_match)}
    
    for CR, CY in zip(cr_list, phred_list):
        if (CR, CY) not in matches:
            matches[(CR, CY)] = '-'
    
    return matches


chunk_size = 10000
corrections = {}

if METHOD == 'edit_distance':
    to_correct = no_whitelist_match.CR.unique()

    while len(to_correct) > 0:
        logging.info('{:,} still to correct\n'.format(len(to_correct)))
        next_batch = to_correct[:min([chunk_size, len(to_correct)])]
        to_correct = [i for i in to_correct if i not in next_batch]
        for cr, correction in fast_correct(next_batch, whitelist).items():
            assert(cr not in corrections)
            corrections[cr] = correction
elif METHOD == 'error_rate':
    to_correct = no_whitelist_match.CR.unique()

    while len(to_correct) > 0:
        logging.info('{:,} still to correct\n'.format(len(to_correct)))
        next_batch = to_correct[:min([chunk_size, len(to_correct)])]
        to_correct = [i for i in to_correct if i not in next_batch]
        for cr, correction in fast_correct_use_error_rate(next_batch, whitelist, umis_per_cb=umi_counts).items():
            assert(cr not in corrections)
            corrections[cr] = correction
elif METHOD == 'phred':
    to_correct = no_whitelist_match[['CR', 'CY']].drop_duplicates()

    while len(to_correct) > 0:
        logging.info('{:,} still to correct\n'.format(len(to_correct)))
        next_batch_cr = to_correct.CR.to_list()[:min([chunk_size, len(to_correct)])]
        next_batch_phred = to_correct.CY.to_list()[:min([chunk_size, len(to_correct)])]
        if len(to_correct) <= chunk_size:
            to_correct = []
        else:
            to_correct = to_correct.iloc[chunk_size:,:]
        for (cr, cy), correction in fast_correct_use_phred(next_batch_cr, next_batch_phred, whitelist, umis_per_cb=umi_counts).items():
            assert((cr, cy) not in corrections)
            corrections[(cr, cy)] = correction



if METHOD == 'error_rate' or METHOD == 'edit_distance':
    whitelist_match['CB'] = whitelist_match.CR
    no_whitelist_match['CB'] = no_whitelist_match.CR.map(corrections)
    corrected = pd.concat([whitelist_match, no_whitelist_match])
elif METHOD == 'phred':
    whitelist_match['CB'] = whitelist_match.CR
    no_whitelist_match['CB'] = [corrections[(CR, CY)] for CR, CY in zip(no_whitelist_match.CR, no_whitelist_match.CY)]
    corrected = pd.concat([whitelist_match, no_whitelist_match])


corrected.to_csv(sys.stdout, sep='\t', index=False)