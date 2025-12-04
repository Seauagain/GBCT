"""
Utilities for `Dataset Module` .
"""
import shutil
import os
import re
import time
import json
import hashlib
import numpy as np
from mpi4py import MPI




def bisectionConcat(index_start, index_end, dim_input, cache_label_path, data_name):
    r"""Concat fragments of dataset by bisection method with compuation cost :math:`O(logN)`. Assume that there is a sequence of datasets in the directory `Cache`: 
    `subdata0.npy`, `subdata1.npy`, ..., `subdata99.npy`, and then **data_name** is `subdata`, **cache_label_path** is `Cache`. 

    Parameters
    ----------
    index_start : int 
        The start index of dataset sequences.
    index_end : int
        The end index of dataset sequences.
    dim_input : int
        The dimension of dataset.
    cache_label_path : str
        The folder containing subdatasets.
    data_name : str
        Suffix of squential `.npy` files. 

    Returns
    -------
    data : numpy.ndarray
        Final dataset concated by all the fragments.

    """
    if index_end-index_start<=1:
        data = np.array([]).reshape(-1, dim_input)
        for idx in range(index_start,index_end+1):
            data_batch_path=os.path.join(cache_label_path,f'{data_name}{idx}.npy')
            data_temp = np.load(data_batch_path)
            data = np.r_[data,data_temp]
        return data
    else:
        mid = index_start + (index_end-index_start)//2
        data_part1=bisectionConcat(index_start,mid,dim_input,cache_label_path,data_name)
        data_part2=bisectionConcat(mid+1,index_end,dim_input,cache_label_path,data_name)
        data = np.r_[data_part1,data_part2]
        return data

def mpiSplitData(sample_num, cpu_id, cpu_size):
    r"""The size will be splitted into **cpu_size** parts and the index of the **cpu_id** part will be returned.

    Parameters
    ----------
    sample_num : int
        The size of given dataset.
    cpu_id : int
        The id of a cpu.
    cpu_size : int
        Total cpu size.
    
    Returns
    -------
    Split_array_index : numpy.ndarray
        The index of the cpu_id-th part.
    
    """
    arr = np.arange(sample_num)
    return np.array_split(arr, cpu_size)[cpu_id]


def allowConcatForMPI(batch_num,cache_path, data_name,time_out=72000):
    r"""Circularly check whether all the sub-datasets have been created. If so, return True.

    Parameters
    ---------
    batch_num : int 
        Total number of sub-datasets expected to be created.
    cache_path : str
        The folder containing all the sub-datasets.
    data_name : str
        Suffix of squential `.npy` files.
    time_out : float,optional
        Max time for waiting sub-dataset being created. Default 20h.

    Returns
    -------
    flag : bool
        Whether all the sub-datasets have been created.
    
    """
    finish = set()
    all_tasks = set(range(batch_num))
    t0 = time.time()
    while len(finish) < batch_num: 
        for idx in all_tasks.difference(finish):
            data_path = os.path.join(cache_path, f'{data_name}{idx}.npy')
            if os.path.exists(data_path):
                finish.add(idx)
        time.sleep(10) #every 10s, check once again
        running_time  = time.time() - t0
        if running_time > time_out:
            print(f'Process terminated. Running time exceeds max allowed time ({time_out}s).')
            return False
    return True


def emptyFolder(target_path, rank=0):
    """
    clear the target directory quickly using `rsync --delete-before -a`.
    e.g. `rsync --delete-before -a __empty__/ DeleteFolder/`
    """
    tmp_home = f"__empty__" # create an empty folder to replace the target path
    os.makedirs(tmp_home, exist_ok=True)
    if os.path.exists(target_path):
        os.system(f"rsync --delete-before -a {tmp_home}/ {target_path}/ 2> /dev/null")
        shutil.rmtree(f"{target_path}", ignore_errors=True)



def mpiClearCache():
    r"""Use MPI parallelization to clear cache folders including CacheManifold*, CacheManifoldBatch*, CacheBatchData* and CacheLabel Data* ."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # print("mpiclear cache: ", rank)
    
    cache_pathes = [f'CacheManifold{0}', f'CacheLabelData{rank}',  f'CacheManifoldBatch{rank}', f'CacheBatchData{rank}']
    for cache_path in cache_pathes:
        try:
            emptyFolder(cache_path, rank=rank)
        except:
            pass
    
    tmp_home = f"__empty__" # create an empty folder to replace the target path
    if os.path.exists(tmp_home) and rank==0:
        shutil.rmtree(f"{tmp_home}", ignore_errors=True) # remove the empty folder created before when rank=0

  
def saveJson(json_path, args: dict):
    r"""Save dict into .json file
    
    Parameters
    ----------
        json_path : str
            The path to save json file.
        args : dict
            Hyper-parameters dict.
    """
    with open(json_path, "w") as f:
        f.write(
            json.dumps(args, ensure_ascii=False, indent=4, separators=(',', ':')))
    f.close()


def checkmd5(file_path):
    r"""get the md5 of specific file"""
    with open(file_path, "rb") as f:
        data = f.read()
    hash_object = hashlib.md5(data)
    hash_hex = hash_object.hexdigest()
    return hash_hex









