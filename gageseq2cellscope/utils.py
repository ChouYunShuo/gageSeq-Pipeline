import numpy as np
import pandas as pd
import h5py
import os
import pickle
import pprint
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

def copyGroup(source_group, dest_group):
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
            copyGroup(item, dest_subgroup)

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

def merge_temp_h5_files(original_h5_path, temp_folder, process_cnt, res):
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
        cell_groups = res_grp["cells"]
        for process_id in range(process_cnt):
            temp_h5_path = os.path.join(temp_folder, f"temp_cells_{process_id}.h5")
            with h5py.File(temp_h5_path, 'r') as temp_hdf:
                for cell_key in temp_hdf.keys():
                    temp_hdf.copy(cell_key, cell_groups)
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
            if name.endswith('/cells'):
                first_child = list(obj.keys())[0] if obj.keys() else None
                if first_child:
                    first_child_path = f"{name}/{first_child}"
                    first_child_obj = obj[first_child]
                    print_attrs(first_child_path, first_child_obj, depth + 1)
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
 
if __name__ == "__main__":
    #file_path = '/work/magroup/yunshuoc/scHDF5_data/4DN_scHi-C_Kim_all.h5'
    file_path = '/work/magroup/yunshuoc/Higashi_Pipeline/4DN_scHi-C_Kim/tmp/chr1_exp1_nbr_0_impute.hdf5'
   
    is_valid, message = check_hdf5_structure(file_path)
    if is_valid:
        print("HDF5 structure is valid.")
    else:
        print(f"Invalid HDF5 structure: {message}")

    #print_hdf5_structure(file_path)

    # check pickle file
    # file_path = '/work/magroup/yunshuoc/Higashi_Pipeline/Ramani_et_al/label_info.pickle'
    # #file_path = '/work/magroup/yunshuoc/Higashi_Pipeline/4DN_scHi-C_Kim/label_info.pickle'  # Adjust the path to your pickle file
    # data = load_pickle(file_path)
    
    # # Use pprint to print the data structure in a readable format
    # for key, value in data.items():
    #     print(f"Key: {key}, Type: {type(value)}")
    # print(len(data["cell type"]))


    # check h5

    # with h5py.File("../../../scHDF5_data/Lee_et_al_001.h5", 'r') as f:
    #     dataset = f["resolutions/500000/cells/cell_1/tracks"]
    #     print(list(dataset.keys()))
        # print(len(f[f'resolutions/100000/bins'].get("chrom")))
        # dataset = f["resolutions/100000/cells/cell_id1/pixels/count"]
        # print(f"Dataset shape: {dataset.shape}")
        # print(f"Dataset size: {dataset.size}")
        # cells = ["cell_0"]