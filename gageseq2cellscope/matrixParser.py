import h5py
import numpy as np
import os
from tqdm import trange
from utils import create_mask

class MatrixParser:
    def __init__(self, impute_dir, raw_dir, chrom_list, chrom_offset, cell_id, neighbors, res, cytoband_path, embedding_name="exp1", process_cnt=1):
        self.impute_dir = impute_dir
        self.raw_dir = raw_dir
        self.chrom_list = chrom_list
        self.chrom_offset = chrom_offset
        self.cell_id = cell_id
        self.neighbors = neighbors
        self.res = res
        self.cytoband_path = cytoband_path
        self.embedding_name = embedding_name
        self.process_cnt = process_cnt

    def __iter__(self):
        if self.process_cnt == 1:
            range_iter = trange(len(self.chrom_list), desc="Processing Chromosomes")
        else:
            range_iter = range(len(self.chrom_list))

        for idx in range_iter:
            chrom = self.chrom_list[idx].decode('utf-8')
            origin_sparse = np.load(os.path.join(self.raw_dir, f"chr{chrom}_sparse_adj.npy"), allow_pickle=True)
            size = origin_sparse[0].shape[0]
            mask = 1 - create_mask(self.res, self.cytoband_path, -1, chrom, origin_sparse)

            with h5py.File(os.path.join(self.impute_dir, f"chr{chrom}_{self.embedding_name}_nbr_{self.neighbors}_impute.hdf5"), "r") as hdf:
                coordinates = hdf['coordinates']
                xs, ys = coordinates[:, 0], coordinates[:, 1]

                proba = np.array(hdf[f"cell_{self.cell_id}"])
                proba = np.maximum(proba - np.min(proba), 0.0)
                proba[proba <= 1e-5] = 0.0

                m = np.zeros((size, size))
                np.add.at(m, (xs, ys), proba)
                np.add.at(m, (ys, xs), proba)
                m *= mask

                proba = m[xs, ys]
                proba1 = m[ys, xs]

                x = np.concatenate([xs, ys]) + self.chrom_offset[idx]
                y = np.concatenate([ys, xs]) + self.chrom_offset[idx]
                proba = np.concatenate([proba, proba1])

                sorter = np.lexsort((y, x))
                x = x[sorter]
                y = y[sorter]
                proba = proba[sorter]

                yield {
                    "bin1_id": x,
                    "bin2_id": y,
                    "count": proba,
                }
            