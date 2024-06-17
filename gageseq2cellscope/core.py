import h5py
import numpy as np
import json
import os
import sys
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm import trange
from collections import defaultdict
from utils import rlencode, get_chroms_from_txt, file_type, merge_temp_h5_files, print_hdf5_structure, check_hdf5_structure
from matrixParser import MatrixParser
import multiprocessing as mp

CHROM_DTYPE = np.dtype("S")
CHROMID_DTYPE = np.int32
CHROMSIZE_DTYPE = np.int32
COORD_DTYPE = np.int32
BIN_DTYPE = np.int64
COUNT_DTYPE = np.float32
OFFSET_DTYPE = np.int64

"""
├── meta
│    ├── label
├── embed
│    ├── pca
│    └── umap
└── resolutions
     ├── 10000
     │   ├── bins
     |   |   ├── chrom
     |   |   ├── start
     |   |   └── end
     │   ├── chroms
     |   |   ├── name
     |   |   └── length
     │   └── cells
     │       ├── cell_1
     │       │   ├── pixels
     |       |   |   ├── bin1_id
     |       |   |   ├── bin2_id
     |       |   |   └── count
     │       │   ├── indexes
     │       │   │   ├── chrom_offset
     |       |   │   └── bin1_offset
     │       │   ├── tracks
     |       |       └── insulation
     │       ├── cell_2
     │       │   ├── pixels
     │       │   └── indexes
     │   └── groups
     │       ├── group_0
     │       │   ├── pixels
     |       |   |   ├── bin1_id
     |       |   |   ├── bin2_id
     |       |   |   └── count
     │       │   ├── indexes
     │       │   │   ├── chrom_offset
     |       |   │   └── bin1_offset
     │       │   ├── tracks
     |       |       └── insulation
     │       ├── group_1
     │       │   ├── pixels
     │       │   └── indexes
     │   
     ├── 50000
     │   

"""

def write_chroms(grp, fname, h5_opts):
    chrom_dict = defaultdict(list)
    all_chrom = get_chroms_from_txt(fname)
    for chrom in all_chrom:
        chrom_dict["name"].append(chrom[0])
        chrom_dict["length"].append(chrom[1])

    n_chroms = len(chrom_dict["name"])

    names = np.array(chrom_dict["name"], dtype=CHROM_DTYPE)
    lengths = np.array(chrom_dict["length"], dtype=CHROMSIZE_DTYPE)

    grp.create_dataset('name', shape=(n_chroms,),
                       dtype=names.dtype, data=names, **h5_opts)
    grp.create_dataset("length", shape=(n_chroms,),
                       dtype=lengths.dtype, data=lengths, **h5_opts)

    return names, lengths

def write_bins(res, chroms_names, chrom_lens, grp, h5_opts):
    bin_dict = defaultdict(list)
    for index in range(len(chroms_names)):
        for i in range(int((chrom_lens[index])/(res))+1):
            bin_dict["chrom"].append(chroms_names[index])
            bin_dict["start"].append(i*res)
            bin_dict["end"].append((i+1)*res)

    n_bins = len(bin_dict["chrom"])

    chroms = np.array(bin_dict["chrom"], dtype=CHROM_DTYPE)
    starts = np.array(bin_dict["start"], dtype=COORD_DTYPE)
    ends = np.array(bin_dict["end"], dtype=COORD_DTYPE)

    grp.create_dataset('chrom', shape=(n_bins,),
                       dtype=chroms.dtype, data=chroms, **h5_opts)
    grp.create_dataset("start", shape=(n_bins,),
                       dtype=starts.dtype, data=starts, **h5_opts)
    grp.create_dataset("end", shape=(n_bins,),
                       dtype=ends.dtype, data=ends, **h5_opts)

def setup_pixels(grp, nbins, h5_opts):
    max_shape = nbins*nbins
    grp.create_dataset('bin1_id', shape=(max_shape,),
                       dtype=BIN_DTYPE, **h5_opts)
    grp.create_dataset("bin2_id", shape=(max_shape,),
                       dtype=BIN_DTYPE, **h5_opts)
    grp.create_dataset("count", shape=(max_shape,),
                       dtype=COUNT_DTYPE, **h5_opts)

def write_pixels(grp, data_folder, contact_map_fname, chrom_list, chrom_offset, cell_id, columns, process_cnt):
    cellMatrixParser = MatrixParser(
         data_folder, contact_map_fname, chrom_list, chrom_offset, cell_id, "2d", process_cnt)
    m_size = 0
    for chunk in cellMatrixParser:
        dsets = [grp[col] for col in columns]
        n = len(chunk[columns[0]])
        for col, dset in zip(columns, dsets):
            dset.resize((m_size + n,))
            dset[m_size: m_size + n] = chunk[col]
        m_size += n

def get_bin_index(grp, n_chroms, n_bins):
    chrom_ids = grp["chrom"]
    chrom_offset = np.zeros(n_chroms + 1, dtype=OFFSET_DTYPE)
    index = 0
    for start, length, value in zip(*rlencode(chrom_ids)):
        chrom_offset[index] = start
        index += 1
    chrom_offset[index] = n_bins

    return chrom_offset

def get_pixel_index(grp, n_bins, n_pixels):
    bin1 = np.array(grp["bin1_id"])
    bin1_offset = np.zeros(n_bins + 1, dtype=OFFSET_DTYPE)
    curr_val = 0

    for start, length, value in zip(*rlencode(bin1, 1000000)):
        bin1_offset[curr_val: value + 1] = start
        curr_val = value+1

    bin1_offset[curr_val:] = n_pixels

    return bin1_offset

def write_index(grp, chrom_offset, bin_offset, h5_opts):
    grp.create_dataset(
        "chrom_offset",
        shape=(len(chrom_offset),),
        dtype=OFFSET_DTYPE,
        data=chrom_offset, **h5_opts
    )
    grp.create_dataset(
        "bin1_offset",
        shape=(len(bin_offset),),
        dtype=OFFSET_DTYPE,
        data=bin_offset, **h5_opts
    )

# append functions
def write_embed(grp, embed, h5_opts):
    vec_pca = PCA(n_components=2).fit_transform(embed)
    vec_umap = UMAP(n_components=2).fit_transform(embed)

    grp.create_dataset("pca", shape=(len(vec_pca),2), data= vec_pca, **h5_opts)
    grp.create_dataset("umap", shape=(len(vec_umap),2), data= vec_umap, **h5_opts)

def write_meta(grp, data, h5_opts):
    vec_label = np.array(data['subclass']).astype(str)
    ascii_label = np.char.encode(vec_label, 'ascii')
    grp.create_dataset("label", shape=(len(ascii_label)), data= ascii_label, **h5_opts)

def write_spatial(grp, data, h5_opts):
    coords = np.column_stack((data['coor_X'].to_numpy(), data['coor_Y'].to_numpy()))
    grp.create_dataset("coords", shape=(len(coords),2), data= coords, **h5_opts)

def write_gene_expr(grp, gene_expr_data, h5_opts):
    gene_expr_data = gene_expr_data.T
    for k in range(gene_expr_data.shape[0]):
        v = gene_expr_data[k, :]
        grp.create_dataset(str(k), shape=(len(v),), data=v, **h5_opts)

def get_track_index(grp, n_bins, n_pixels):
    bin = np.array(grp["bin_id"])
    bin_offset = np.zeros(n_bins + 1, dtype=OFFSET_DTYPE)
    curr_val = 0

    for start, length, value in zip(*rlencode(bin, 1000000)):
        bin_offset[curr_val: value + 1] = start
        curr_val = value+1

    bin_offset[curr_val:] = n_pixels

    return bin_offset

def setup_track(grp, nbins, h5_opts):
    max_shape = 2*nbins
    grp.create_dataset('bin_id', shape=(max_shape,),
                       dtype=BIN_DTYPE, **h5_opts)
    grp.create_dataset("count", shape=(max_shape,),
                       dtype=COUNT_DTYPE, **h5_opts)

def createTrack(data_folder, cur_grp, track_type: str, track_f_name:str, chrom_list, cur_res: int, n_bins, cell_id, h5_opts):
    if track_type in cur_grp:
        del cur_grp[track_type]

    track_dataset = cur_grp.create_dataset(track_type, shape=(2*n_bins,),
                       dtype=COUNT_DTYPE, **h5_opts)
    chrom_offset = cur_grp.parent["indexes"].get("chrom_offset")
    cellMatrixParser = MatrixParser(
         data_folder, track_f_name, chrom_list, chrom_offset, cell_id, "1d", 1, track_type)
    m_size = 0
    for chunk in cellMatrixParser:
        n = len(chunk[track_type])
        track_dataset.resize((m_size + n,))
        track_dataset[m_size: m_size + n] = chunk[track_type]
        m_size += n


### functions for parallel process contact maps
def process_cells_range(start, end, process_id, temp_folder, data_folder, contact_map_file_name, np_chroms_names, chrom_offset, h5_opts, n_bins, progress, process_cnt):
    temp_h5_path = os.path.join(temp_folder, f"temp_cells_{process_id}.h5")
    with h5py.File(temp_h5_path, 'w') as hdf:
        for i in range(start, end):
            print(f"Process {process_id} processing cell {i}")
            cur_cell_grp = hdf.create_group(f"cell_{i}")
            cell_grp_pixels = cur_cell_grp.create_group("pixels")
            setup_pixels(cell_grp_pixels, n_bins, h5_opts)
            write_pixels(cell_grp_pixels,  data_folder, contact_map_file_name,  np_chroms_names,
                             chrom_offset, i, list(cur_cell_grp["pixels"]), process_cnt)
            n_pixels = len(cur_cell_grp["pixels"].get("bin1_id"))
            bin_offset = get_pixel_index(cur_cell_grp["pixels"], n_bins, n_pixels)
            grp_index = cur_cell_grp.create_group("indexes")
            write_index(grp_index, chrom_offset, bin_offset, h5_opts)

            with progress.get_lock():
                progress.value += 1
                if progress.value % 10 == 0:  # Print progress every 10 cells
                    print(f"Process {process_id} has completed {progress.value} cells")

class SCHiCGenerator:
    def __init__(self, config_path):
        self.data_folder = ""
        self.output_path = ""
        self.contact_map_file_name = "" 
        self.embed_file_name = ""
        self.meta_file_name = ""
        self.gene_expr_file_name = ""
        self.tracks = [] # store the 1d tracks name and its file name
        self.chrom_size_path = ""
        self.resolutions = []
        self.cell_cnt = 0
        self.h5_opts = {}
        self.process_cnt = 0
        self.load_base_config(config_path)
    
    def load_base_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        
            for key, default in self.__dict__.items():
                if not key.startswith('_'):
                    setattr(self, key, config.get(key, default))
            print("Config loaded")
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {config_path}")

    def create_all_h5(self):
        try:
            self.validate_paths()
            print("all data folders and files validated")
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error: {e}")
            os.remove(self.output_path)
            sys.exit(1)
        with h5py.File(self.output_path, 'w') as hdf:
            hdf.create_group("resolutions")

        for res in self.resolutions:
            print("Creating resolution: "+str(res))
            self.create_res_h5(res) 

    def create_res_h5(self, res):
        with h5py.File(self.output_path, 'r+') as hdf:
            res_grp = hdf.create_group("resolutions/"+str(res))

            # write res/chrom
            grp_chroms = res_grp.create_group("chroms")
            np_chroms_names, np_chroms_length = write_chroms(
                grp_chroms, self.chrom_size_path, self.h5_opts)
            
            # write res/bin
            grp_bins = res_grp.create_group("bins")
            write_bins(res, np_chroms_names,
                       np_chroms_length, grp_bins, self.h5_opts)
            cell_groups = res_grp.create_group("cells")

            n_bins = len(res_grp["bins"].get("chrom"))
            n_chroms = len(res_grp["chroms"].get("length"))
            chrom_offset = get_bin_index(
                    res_grp["bins"], n_chroms, n_bins)
        
            if self.process_cnt == 1:
                self.process_cells(cell_groups, n_bins, np_chroms_names, chrom_offset)
                return
        if self.process_cnt > 1:
            self.parallel_process_cells(np_chroms_names, chrom_offset, n_bins, res)
    
    def parallel_process_cells(self, np_chroms_names, chrom_offset, n_bins, res):
        # Determine the range of cells each process should handle
        cells_per_process = (self.cell_cnt + self.process_cnt - 1) // self.process_cnt
        temp_folder = "/scratch/tmp-yunshuo/temp_h5"
        os.makedirs(temp_folder, exist_ok=True)
        
        # Create a shared progress counter
        progress = mp.Value('i', 0)
        
        # Create a pool of worker processes
        processes = []
        for process_id in range(self.process_cnt):
            start = process_id * cells_per_process
            end = min((process_id + 1) * cells_per_process, self.cell_cnt)
            if start < end:
                p = mp.Process(target=process_cells_range, args=(start, end, process_id, temp_folder, self.data_folder, self.contact_map_file_name, np_chroms_names, chrom_offset, self.h5_opts, n_bins, progress, self.process_cnt))
                processes.append(p)
                p.start()
        print("in parallel_process_cells 2")
        # Wait for all processes to finish
        for p in processes:
            p.join()
        # Merge the temporary HDF5 files into the original HDF5 file
        merge_temp_h5_files(self.output_path, temp_folder, self.process_cnt, res)

    def process_cells(self, cell_groups, n_bins, np_chroms_names, chrom_offset):
        for i in range(self.cell_cnt):
            print("cell "+str(i)+":")
            cur_cell_grp = cell_groups.create_group(
                "cell_"+str(i))
            
            cell_grp_pixels = cur_cell_grp.create_group("pixels")
            setup_pixels(cell_grp_pixels, n_bins, self.h5_opts)
            write_pixels(cell_grp_pixels,  self.data_folder,  self.contact_map_file_name,  np_chroms_names,
                            chrom_offset, i, list(cur_cell_grp["pixels"]), self.process_cnt)
            n_pixels = len(cur_cell_grp["pixels"].get("bin1_id"))
            
            # write index for each cell
            bin_offset = get_pixel_index(
                cur_cell_grp["pixels"], n_bins, n_pixels)
            grp_index = cur_cell_grp.create_group("indexes")
            write_index(grp_index, chrom_offset, bin_offset, self.h5_opts)
    
    def append_h5(self, atype: str):
        if not os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " not exists")

        if(atype =='embed'):
            print("appending cell embeddings...")
            embed_path = os.path.join(self.data_folder, self.embed_file_name)
            if not os.path.exists(embed_path):
                raise RuntimeError("File: " + self.embed_file_name + " does not exists, you need to load it first!")
            if file_type(embed_path) == 'na':
                raise RuntimeError("Cannot recognize file: " + embed_path + "'s file type!")

            if file_type(embed_path) == 'npy':
                cell_embeddings = np.load(embed_path)
            elif file_type(embed_path) == 'pkl':
                cell_embeddings = pickle.load(open(embed_path, "rb"))

            with h5py.File(self.output_path, 'a') as hdf:
                if 'embed' in hdf:
                    # Delete the group 'rgrp'
                    del hdf['embed']
                emb_grp = hdf.create_group('embed')
                write_embed(emb_grp, cell_embeddings, self.h5_opts)
        
        elif(atype == 'meta'):
            print("appending cell meta data...")
            meta_path = os.path.join(self.data_folder, self.meta_file_name)
            if not os.path.exists(meta_path):
                raise RuntimeError("File: " + self.meta_file_name + " does not exists, you need to load it first!")
            if file_type(meta_path) == 'na':
                raise RuntimeError("Cannot recognize file: " + meta_path + "'s file type!")
            
            data = pd.read_csv(meta_path)
            with h5py.File(self.output_path, 'a') as hdf:
                if 'meta' in hdf:
                    # Delete the group 'rgrp'
                    del hdf['meta']
                meta_grp = hdf.create_group('meta')
                write_meta(meta_grp, data, self.h5_opts)

        elif(atype == 'spatial'):
            print("appending cell spatial data...")
            spatial_path = os.path.join(self.data_folder, self.meta_file_name)
            if not os.path.exists(spatial_path):
                raise RuntimeError("File: " + self.meta_file_name + " does not exists, you need to load it first!")
            if file_type(spatial_path) == 'na':
                raise RuntimeError("Cannot recognize file: " + spatial_path + "'s file type!")
            
            data = pd.read_csv(spatial_path)
            with h5py.File(self.output_path, 'a') as hdf:
                if 'spatial' in hdf:
                    # Delete the group 'rgrp'
                    del hdf['spatial']
                spatial_grp = hdf.create_group('spatial')
                write_spatial(spatial_grp, data, self.h5_opts)

        elif(atype == 'gene_expr'):
            print("appending gene_expr data...")
            gene_path = os.path.join(self.data_folder, self.gene_expr_file_name)
            if not os.path.exists(gene_path):
                raise RuntimeError("File: " + self.gene_expr_file_name + " does not exists, you need to load it first!")
            if file_type(gene_path) == 'na':
                raise RuntimeError("Cannot recognize file: " + gene_path + "'s file type!")

            if file_type(gene_path) == 'npy':
                gene_expr_mat = np.load(gene_path)
            elif file_type(gene_path) == 'pkl':
                gene_expr_mat = pickle.load(open(gene_path, "rb"))

            with h5py.File(self.output_path, 'a') as hdf:
                if 'gene_expr' in hdf:
                    # Delete the group 'rgrp'
                    del hdf['gene_expr']
                gene_grp = hdf.create_group('gene_expr')
                write_gene_expr(gene_grp, gene_expr_mat, self.h5_opts)

        elif(atype=='1dtrack'):
            print("appending 1d track data...")
            with h5py.File(self.output_path, 'a') as hdf:
                for res_file in self.tracks:
                    cur_res = res_file["resolution"]
                    n_bins = len(hdf[f'resolutions/{cur_res}/bins'].get("chrom"))
                    chrom_list = hdf[f'resolutions/{cur_res}/chroms'].get("name")

                    # Pre-calculate group paths to reduce file navigation in the inner loop
                    cell_groups = [f"resolutions/{cur_res}/cells/cell_{i}" for i in range(self.cell_cnt)]
                    range_iter = trange(len(cell_groups), desc="Processing Cell")
                    for cell_id in range_iter:
                        cell_group = cell_groups[cell_id]
                        cur_grp = hdf[cell_group]
                        if "tracks" in cur_grp:
                            del cur_grp["tracks"]
                        track_grp = cur_grp.create_group('tracks')

                        for track_type, track_f_name in res_file["track_object"].items():   
                            createTrack(self.data_folder, track_grp, track_type, track_f_name, chrom_list, cur_res, n_bins, cell_id, self.h5_opts)
        else:
            raise ValueError("Invalid atype provided. Only 'embed, meta, 1dtrack' is supported.")

    def validate_paths(self):
        if os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " already exists")

        embed_file = os.path.join(self.data_folder, self.embed_file_name)
        if not os.path.exists(embed_file):
            raise FileNotFoundError(f"The file '{embed_file}' does not exist.")

        meta_file = os.path.join(self.data_folder, self.meta_file_name)
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"The file '{meta_file}' does not exist.") 

        if not os.path.exists(self.chrom_size_path):
            raise FileNotFoundError(f"The file '{self.chrom_size_path}' does not exist.")
        

    def print_schema(self):
        if not os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " does not exist")
        print_hdf5_structure(self.output_path)
    
    def check_schema(self):
        if not os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " does not exist")
        is_valid, message = check_hdf5_structure(self.output_path)
        if is_valid:
            print("HDF5 structure is valid.")
        else:
            print(f"Invalid HDF5 structure: {message}")

def generate_hic_file(config_path, mode, types=[]):
    generator = SCHiCGenerator(config_path)
    if mode == 'create':
        generator.create_all_h5()
    elif mode == 'append':
        for t in types:
            generator.append_h5(t)
    elif mode == 'print':
        generator.print_schema()
    elif mode == 'check':
        generator.check_schema()

if __name__ == "__main__":
    generator =  SCHiCGenerator("../config_cluster.JSON")
    # try:
    generator.create_all_h5()
    generator.append_h5("1dtrack")
    generator.append_h5("embed")
    generator.append_h5("meta")
    generator.append_h5("spatial")
    generator.append_h5("gene_expr")
    # except Exception as e:
    #     print(repr(e))
    #     os.remove(generator.output_path)
