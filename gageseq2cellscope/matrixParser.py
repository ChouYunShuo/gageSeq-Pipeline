import h5py
import numpy as np
import os
from tqdm import trange
from utils import create_mask
import pickle

class MatrixParser:
    def __init__(self, data_folder, contact_map_fname, chrom_list, chrom_offset, cell_id, mat_type, process_cnt, track_type = ""):
        self.fname = os.path.join(data_folder, contact_map_fname)
        self.chrom_list = chrom_list
        self.chrom_offset = chrom_offset
        self.cell_id = cell_id
        self.type = mat_type
        self.track_type = mat_type if track_type == "" else track_type

    def parse_2d_matrix(self, file_path, idx):
        with open(file_path, 'rb') as f:
            coo_matrix = pickle.load(f)[self.cell_id]
            xs, ys, proba = coo_matrix.row, coo_matrix.col, coo_matrix.data
            
            # Process probability values
            proba = self.process_proba(proba)
            x, y = self.adjust_indices(xs, ys, idx)

            # Sort indices and proba together
            sorter = np.lexsort((y, x))
            x, y, proba = x[sorter], y[sorter], proba[sorter]
            
            return x, y, proba

    def parse_1d_track(self, file_path, idx):
        with open(file_path, 'rb') as f:
            track_data = pickle.load(f)[self.cell_id]
            proba = self.process_proba(np.array(track_data), '1d')
            return proba

    def process_proba(self, proba, mat_type='2d'):
        proba /= np.sum(proba)
        proba[proba <= 1e-6] = 0.0
        proba *= 1e6
        return np.concatenate([proba, proba]) if mat_type == '2d' else proba

    def adjust_indices(self, xs, ys, idx):
        x = np.concatenate([xs, ys]) + self.chrom_offset[idx]
        y = np.concatenate([ys, xs]) + self.chrom_offset[idx]
        return x, y

    def __iter__(self):   
        if self.process_cnt == 1:
            range_iter = trange(len(self.chrom_list), desc="Processing Chromosomes")
        else:
            range_iter = range(len(self.chrom_list))

        for idx in range_iter:
            chrom = self.chrom_list[idx].decode('utf-8')
            file_path = self.fname + chrom + ".pkl"
            if(self.type == '1d'):
                proba = self.parse_1d_track(file_path ,idx)
                yield {
                    self.track_type: proba,
                }
            elif(self.type == '2d'):
                x, y, proba = self.parse_2d_matrix(file_path ,idx)
                yield {
                    "bin1_id": x,
                    "bin2_id": y,
                    "count": proba,
                }