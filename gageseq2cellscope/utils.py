import numpy as np
import pandas as pd
import h5py
import os
import pickle
from collections import defaultdict
import urllib.parse
OFFSET_DTYPE = np.int64


def rlencode(array, chunksize=None):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    array = np.array(array)
    n = len(array)
    if n == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=array.dtype),
        )

    if chunksize is None:
        chunksize = n

    starts, values = [], []
    last_val = np.nan
    for i in range(0, n, chunksize):
        x = array[i: i + chunksize]
        locs = where(x[1:] != x[:-1]) + 1
        if x[0] != last_val:
            locs = np.r_[0, locs]
        starts.append(i + locs)
        values.append(x[locs])
        last_val = x[-1]
    starts = np.concatenate(starts)
    lengths = np.diff(np.r_[starts, n])
    values = np.concatenate(values)

    return starts, lengths, values


def create_mask(res, c_path, k=30, chrom="chr1", origin_sparse=None):
    """
    Create a mask for the given resolution and chromosome.

    Parameters
    ----------
    res : int
        Resolution of the mask.
    c_path : str
        Path to the cytoband file.
    k : int, optional
        Size of the diagonal band to mask out.
    chrom : str, optional
        Chromosome name.
    origin_sparse : array_like, optional
        Sparse array representing the original data.

    Returns
    -------
    array
        The generated mask.
    """
    final = np.array(np.sum(origin_sparse, axis=0).todense())
    size = origin_sparse[0].shape[-1]
    a = np.zeros((size, size))
    if k > 0:
        for i in range(min(k, len(a))):
            for j in range(len(a) - i):
                a[j, j + i] = 1
                a[j + i, j] = 1
        a = np.ones_like((a)) - a
	
    gap = np.sum(final, axis=-1, keepdims=False) == 0

    gap_tab = pd.read_table(c_path, sep="\t", header=None)
    gap_tab.columns = ['chrom', 'start', 'end', 'sth', 'type']
    gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
    start = np.floor((np.array(gap_list['start'])) / res).astype('int')
    end = np.ceil((np.array(gap_list['end'])) / res).astype('int')

    for s, e in zip(start, end):
        a[s:e, :] = 1
        a[:, s:e] = 1
    a[gap, :] = 1
    a[:, gap] = 1

    return a

def sort_key(item):
    name = item[0]
    if name.isdigit():
        return int(name)
# FIXME: add X,Y, consider chr and idx representations 
def get_chroms_from_txt(filename):
    chromosome_data = []
    
    with open(filename, 'r') as file:
        for line in file:
            name, length = line.split()
            if name.startswith("chr") and (name[3:].isdigit()):#or name[3] in ["X"]):
                chromosome_data.append((name[3:], int(length)))

    chromosome_data.sort(key=sort_key)
    return chromosome_data

def copy_group(source_group, dest_group):
    """
    Recursively copy the contents of a source HDF5 group to a destination group.

    Parameters
    ----------
    source_group : h5py.Group
        The source group to copy from.
    dest_group : h5py.Group
        The destination group to copy to.
    
    """
    for key, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            dest_group.create_dataset(key, data=item[...], dtype=item.dtype, shape=item.shape)
        elif isinstance(item, h5py.Group):
            dest_subgroup = dest_group.create_group(key)
            copy_group(item, dest_subgroup)

def copy_dataset(item, dest_group, track_type):
    """
    Copy a single HDF5 dataset to a destination group with a specified track type.

    Parameters
    ----------
    item : h5py.Dataset
        The dataset to copy.
    dest_group : h5py.Group
        The destination group to copy to.
    track_type : str
        The name to give the copied dataset in the destination group.
    
    """
    if isinstance(item, h5py.Dataset):
        dest_group.create_dataset(track_type, data=item[...], dtype=item.dtype, shape=item.shape)
        

def file_type(fname: str):
    """
    Determine the file type based on the file extension.

    Parameters
    ----------
    fname : str
        The name of the file.
    Returns
    -------
    str
        The determined file type ('npy', 'pkl', 'csv', or 'na' for unknown).
    """
    return 'npy' if fname.endswith('npy') else 'pkl' if fname.endswith('pkl') else 'pkl' if fname.endswith('pickle') else 'csv' if fname.endswith('csv') else 'na'

def merge_temp_h5_files(original_h5_path, temp_folder, process_cnt, res, neighbor_num=[0]):
    """
    Merge temporary HDF5 files into an original HDF5 file.

    Parameters
    ----------
    original_h5_path : str
        Path to the original HDF5 file.
    temp_folder : str
        Directory containing the temporary HDF5 files.
    process_cnt : int
        Number of temporary files to merge.
    res : int
        Resolution to use for merging.
    
    """
    with h5py.File(original_h5_path, 'a') as original_hdf:
        res_grp = original_hdf["resolutions"][str(res)]
        layer_groups = res_grp["layers"]
        imputed_grps = {}
        for num in neighbor_num:
            imputed_grps[num] = layer_groups.create_group(f"imputed_{num}neighbor")
        for process_id in range(process_cnt):
            temp_h5_path = os.path.join(temp_folder, f"temp_cells_{process_id}.h5")
            with h5py.File(temp_h5_path, 'r') as temp_hdf:
                for num in neighbor_num:
                    for cell_key in temp_hdf[f"imputed_{num}neighbor"].keys():
                        temp_hdf.copy(f"imputed_{num}neighbor/{cell_key}", imputed_grps[num])
                
            os.remove(temp_h5_path)  # Remove temporary file after merging

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def print_hdf5_structure(file_path):
    """
    Print the structure of an HDF5 file, including groups and datasets, with
    indentation to represent depth in the hierarchy. Recursively prints only
    the first child of the 'cells' group.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    """
    def print_attrs(name, obj, depth=0):
        padding = ' ' * (depth * 4)  # 4 spaces for each level of depth
        if isinstance(obj, h5py.Group):
            print(f"{padding}Group: {name}")
            if name.endswith('neighbor') or name.endswith('raw') or name.endswith("score") or name.endswith("gene_expr"):
                first_child = list(obj.keys())[0] if obj.keys() else None
                if first_child:
                    first_child_path = f"{name}/{first_child}"
                    first_child_obj = obj[first_child]
                    print_attrs(first_child_path, first_child_obj, depth + 1)
                # if "group" in obj.keys():
                #     grp_path = f"{name}/group"
                #     print_attrs(grp_path, obj["group"], depth + 1)
            else:
                for key in obj.keys():
                    print_attrs(f"{name}/{key}", obj[key], depth + 1)
        elif isinstance(obj, h5py.Dataset):
            print(f"{padding}Dataset: {name}")
            print(f"{padding}    Shape: {obj.shape}")
            print(f"{padding}    Data type: {obj.dtype}")

    with h5py.File(file_path, 'r') as file:
        print_attrs('', file['/'])


def check_hdf5_structure(file_path):
    required_embed_meta_datasets = ['/embed/pca', '/embed/umap','/meta/label']
    
    required_resolution_datasets = [
        'bins/chrom', 'bins/start','bins/end',
        'chroms/length', 'chroms/name',
        'cells/cell_0/indexes/bin1_offset', 'cells/cell_0/indexes/chrom_offset',
        'cells/cell_0/pixels/bin1_id', 'cells/cell_0/pixels/bin2_id', 'cells/cell_0/pixels/count'
    ]
    
    def check_datasets(group, datasets):
        for ds in datasets:
            if ds not in group:
                return False, f"Missing dataset {ds}"
        return True, None

    with h5py.File(file_path, 'r') as f:
        if '/embed' in f and '/meta' not in f:
            return False, "Missing /meta group"
        if '/meta' in f and '/embed' not in f:
            return False, "Missing /embed group"

        if '/embed' in f and '/meta' in f:
            result, message = check_datasets(f, required_embed_meta_datasets)
            if not result:
                return False, message

        if '/resolutions' not in f:
            return False, "Missing /resolutions group"
        
        for resolution in f['/resolutions']:
            resolution_group = f['/resolutions'][resolution]
            for required_ds in required_resolution_datasets:
                if required_ds not in resolution_group:
                    return False, f"Missing dataset {required_ds} in {resolution}"

    return True, "HDF5 structure is valid"

def get_celltype_dict(file_path,label_name):
    data = pd.read_csv(file_path)
    cellTypeDict = defaultdict(list)

    for idx, type in enumerate(data[label_name]):
        cellTypeDict[type].append(idx)

    return cellTypeDict

def get_chr_size_list(dset, res):
    return [2*d//res for d in dset]

def encode_name(name):
    return urllib.parse.quote(name, safe='')

# Function to decode group names
def decode_name(encoded_name):
    return urllib.parse.unquote(encoded_name)

if __name__ == "__main__":

    # check cell type dict 
    #file_path = "/work/magroup/tianming/Researches/sc-hic/data2/final/results_mBC_spatial_full_data/meta_mouse2_slice99.csv"
    # cellTypeDict = get_celltype_dict(file_path, "subclass")
    
    # check chr size list
    
    # file_path = "/work/magroup/tianming/Researches/sc-hic/data2/final/results_mBC_spatial_full_data/expression_mouse1_slice122.pkl"
    # data = load_pickle(file_path)
    # print(data.shape)

    # with h5py.File(file_path, 'r') as hdf: 
    #     res = 100000
    #     grp = hdf[f"resolutions/{res}/gene_expr"]
    #     print(len(grp.keys()))
    
    file_path = "/work/magroup/tianming/Researches/sc-hic/data2/final/results_mBC_spatial_full_data/genes.csv"
    df = pd.read_csv(file_path)

    print(df.columns.tolist())
    print(df['gene_symbol'][0])