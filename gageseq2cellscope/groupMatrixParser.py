import h5py
import numpy as np
import os
from tqdm import trange
from utils import create_mask
import multiprocessing as mp
import pickle

class GroupMatrixParser:
    def __init__(self, data_folder, contact_map_fname, chrom_list, chrom_offset, cell_ids, mat_type, process_cnt, chrom_sizes):
        self.fname = os.path.join(data_folder, contact_map_fname)
        self.chrom_list = chrom_list
        self.chrom_offset = chrom_offset
        self.cell_ids = cell_ids
        self.type = mat_type
        self.process_cnt = process_cnt
        self.chrom_sizes = chrom_sizes

    def parse_2d_matrix(self, q, data, start, end, size):
        m = np.zeros((size, size))
        for cell_id in range(start, end):
            coo_matrix = data[cell_id]
            xs, ys, proba = coo_matrix.row, coo_matrix.col, coo_matrix.data
            proba = self.process_proba(proba)
            np.add.at(m, (xs, ys), proba)
            np.add.at(m, (ys, xs), proba)
        q.put(m)
        # for cell_id in range(start, end):
        #     coo_matrix = pickle.load(file_path)[cell_id]
        #     xs, ys, proba = coo_matrix.row, coo_matrix.col, coo_matrix.data
            
        #     # Process probability values
        #     proba = self.process_proba(proba)

        #     # Sort indices and proba together
        #     sorter = np.lexsort((y, x))
        #     x, y, proba = x[sorter], y[sorter], proba[sorter]
            
        #     return x, y, proba

    def parse_1d_track(self, file_path, idx):
        with open(file_path, 'rb') as f:
            track_data = pickle.load(f)[self.cell_id]
            proba = self.process_proba(np.array(track_data), '1d')
            return proba

    def process_proba(self, proba, mat_type='2d'):
        proba /= np.sum(proba)
        proba[proba <= 1e-6] = 0.0
        proba *= 1e6
        return proba

    def adjust_indices(self, xs, ys, idx):
        x = np.concatenate([xs, ys]) + self.chrom_offset[idx]
        y = np.concatenate([ys, xs]) + self.chrom_offset[idx]
        return x, y

    def __iter__(self):   
        range_iter = trange(len(self.chrom_list), desc="Processing Chromosomes")
        for idx in range_iter:
            chrom = self.chrom_list[idx].decode('utf-8')
            file_path = self.fname + chrom + ".pkl"
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            size = self.chrom_sizes[idx]

            m = np.zeros((size, size))
            q = mp.Queue()
            processes = []
            cells_per_process = len(self.cell_ids) // self.process_cnt
            for process_id in range(self.process_cnt):
                start = process_id * cells_per_process
                end = min((process_id + 1) * cells_per_process, len(self.cell_ids))
                if start < end:
                    p = mp.Process(target=self.parse_2d_matrix, args=(q, data, start, end, size))
                    processes.append(p)
                    p.start()

            while not q.empty():
                m += q.get()

            for p in processes:
                p.join()

            xs, ys = np.nonzero(m)
            proba = m[xs, ys]
            proba1 = m[ys, xs]

            x, y = self.adjust_indices(xs,ys, idx) 
            proba = np.concatenate([proba, proba1])

            sorter = np.lexsort((y, x))
            x, y, proba = x[sorter], y[sorter], proba[sorter]
            
            yield {
                "bin1_id": x,
                "bin2_id": y,
                "count": proba,
            }