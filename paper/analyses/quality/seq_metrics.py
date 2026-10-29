"""Sequence metrics of the steered generation analysis: mean Hamming distance between IMGT-aligned sequences (binarised
one-hot encoding, sequences padded with stops; protein alphabet in the order of MuE) and diversity, the nearest-neighbour
cosine distance between 3-mer count vectors."""
import logging
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

PROT = np.array(["R", "H", "K", "D", "E", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "]"])


def get_ohe(seqs, include_stop=False, fill_stop=False):
    """One-hot encodings [..., length, alphabet] of string sequences; with fill_stop, every entry that is no letter of the
    alphabet (padding, gaps) becomes a stop."""
    alphabet = PROT[:len(PROT) - 1 + include_stop]
    flat = np.array(seqs).flatten()
    max_len = max(len(s) for s in flat)
    ohe = np.array([np.pad(list(s), (0, max_len - len(s))) for s in flat])[..., None] == alphabet
    if fill_stop:
        ohe[..., -1] += (np.sum(ohe, axis=-1) == 0)
    return ohe.reshape(np.shape(seqs) + (max_len, -1))


def get_binarize(seqs, include_stop=False, fill_stop=False):
    ohe = get_ohe(seqs, include_stop=include_stop, fill_stop=fill_stop)
    return np.dot(ohe, 2 ** (np.arange(np.shape(ohe)[-1]) + 1) - 1).astype(int) + 1


def pad_axis(seqs, pad_to_len, axis, pad_val):
    """Pad (or cut) axis of seqs to length pad_to_len with pad_val (NaN where the rest of the axis is NaN)."""
    shape = list(seqs.shape)
    if shape[axis] >= pad_to_len:
        index = [np.s_[:]] * len(seqs.shape)
        index[axis] = np.s_[pad_to_len:]
        if seqs[tuple(index)].sum() > 0:
            logging.warning("Sequence info lost by pad_seq_len.")
        index[axis] = np.s_[:pad_to_len]
        return seqs[tuple(index)]
    broadcast_index = [np.s_[:]] * len(seqs.shape)
    broadcast_index[axis] = np.s_[None]
    shape[axis] = pad_to_len - shape[axis]
    pad = pad_val * np.ones(shape, dtype=seqs.dtype)
    axis_is_nan = np.all(np.isnan(seqs), axis=axis)[tuple(broadcast_index)].repeat(shape[axis], axis=axis)
    if np.any(axis_is_nan):
        pad[axis_is_nan] = np.nan
    return np.concatenate([seqs, pad], axis=axis)


def hamming_dist(seqs_x, seqs_y):
    """Hamming distances [len(seqs_x), len(seqs_y)] between aligned protein sequences, as if every sequence ended
    with infinitely many stops."""
    bin_x = get_binarize(seqs_x, include_stop=True, fill_stop=True)
    bin_y = get_binarize(seqs_y, include_stop=True, fill_stop=True)
    shape_x, shape_y = np.shape(bin_x)[:-1], np.shape(bin_y)[:-1]
    lens = np.max([np.shape(bin_x)[-1], np.shape(bin_y)[-1]])
    bin_x = pad_axis(bin_x, lens, -1, 1).reshape([-1, lens])
    bin_y = pad_axis(bin_y, lens, -1, 1).reshape([-1, lens])
    h_dist = np.empty([len(bin_x), len(bin_y)])
    num_batches = np.ceil(len(bin_y) * len(bin_x) * lens / 1e8).astype(int)   # at most 1e8 entries per batch
    for batch_inds in np.array_split(np.arange(len(bin_x)), num_batches, axis=0):
        sims = np.bitwise_and(np.tile(bin_x[batch_inds][:, None, :], (1, len(bin_y), 1)),
                              np.tile(bin_y[None, :, :], (len(batch_inds), 1, 1))) > 0
        h_dist[batch_inds] = np.sum((1 - sims) >= 1, axis=-1)
    return h_dist.reshape(shape_x + shape_y)


def create_kmers(sequence, k):
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def compute_pairwise_cosine_distances(sequences: List[str], k=3):
    """Per sequence, the cosine distance between its k-mer counts and those of its nearest other sequence."""
    kmers = [" ".join(create_kmers(seq, k)) for seq in sequences]
    m = CountVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"\S+").fit_transform(kmers)
    d = cosine_distances(m)
    return np.min(d + np.eye(d.shape[0]), axis=1)
