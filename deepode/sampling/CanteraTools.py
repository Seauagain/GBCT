## system import 
import cantera as ct
import numpy as np
from multiprocessing import Pool
import time
import os, sys, re
from math import ceil, floor
import shutil

try:
    from mpi4py import MPI
except:
    print('Warning: no module named mpi4py')

## custom import 
from .utils_data import *


def ctOneStep(state, gas_config, delta_t, advance_type):
    r"""Use Cantera to advance the state of the reactor network from the current time :math:`t` towards :math:`t+\Delta t`.

    Parameters
    ----------
    state : numpy.ndarray
        State vector organized as T,P(atm),Y.
    gas_config : list or tuple
        Basic configurations for gas reactor. Organized as [fuel, reactor, cantera_max_time_step]. 
        **fuel** (str): fuel species name.  **reactor** (str): reactor type, 'constP' or 'constV'.  **cantera_max_time_step** (float): max time step allowed in CVODE.
    delta_t : float
        Evolution time step (sec).
    advance_type : str
        The type of advancing for GMS (`label` or `evolution`). Type `evolution` can be more tolerant for  calculation error but expect fewer computation cost. 
    
    Returns
    -------
    state_out : numpy.ndarray
        The output state vector after :math:`\Delta t` which is organized as T,P(atm),Y.

    """
    fuel, reactor, Cantera_Built_in_Delta_t = gas_config
    n = gas.n_species  #gas species
    state = state.reshape(-1, n + 2)
    gas.TPY = state[0, 0], state[0, 1] * ct.one_atm, state[0, 2:]
    reactor_types = {
        'constV': ct.IdealGasReactor,
        'constP': ct.IdealGasConstPressureReactor
    }
    r = reactor_types[reactor](gas)
    sim = ct.ReactorNet([r])
    # sim.reinitialize()
    if advance_type == 'label':
        sim.max_time_step = Cantera_Built_in_Delta_t  # max time step for CVODE
        # sim.atol = 1e-25  # default 1e-15
        sim.rtol = 1e-15  # default 1e-9
    elif advance_type == 'evolution':  #allow
        sim.rtol = 1e-15  # default 1e-9
    else:
        raise ValueError(
            f"advance_type should be 'label' or 'evolution' but got {advance_type}"
        )
    sim.max_steps = 6e6  #default 2E4
    sim.advance(delta_t)
    state_out = np.hstack(gas.TPY)
    state_out[1] = gas.P / ct.one_atm
    return state_out.reshape(1, -1)


def _checkDataPath(data_path, threshold=0):
    r"""Data exists and elements>=threshold, return True. Threshold should be a non-positive value.

    Parameters
    ----------
    data_path : str
        The path where data locates.
    threshold : float
        A non-positive value.
    
    Returns
    -------
    flag : bool
        Whether data exists and elements>= **threshold** .
    
    """
    if not os.path.exists(data_path):  # data doesn't exist
        return False
    else:
        temp = np.load(data_path)
        zero = np.where(temp < threshold)[0]  # zero.size=0 return True
        return zero.size == 0


def _ctOneStepWrapper(cache_batch_path, input_batch_path, batch_index,
                      gas_config, delta_t, threshold, row_in_each_bacth,
                      advance_type, save_data = True):
    r"""This a runner for ``multi-processing``. Load the input batch and then use Cantera to calculate the label of a choosen index in input batch. The label will be saved in **cache_batch_path**.
    If Cantera encounters an unexpected error and does not return a valid value, then an all-zero array will be saved instead to ensure that each input bacth and corresponding label batch have the same size.

    Parameters
    ----------
    cache_batch_path : str
        Path where to save the label.
    input_batch_path : str
        The input bath path.
    batch_index : str
        The batch index of an input batch since the whole input will be splitted in to batches.
    gas_config : list or tuple
        Basic configurations for gas reactor. Organized as [fuel, reactor, cantera_max_time_step]. 
        **fuel** (str): fuel species name.  **reactor** (str): reactor type, 'constP' or 'constV'.  **cantera_max_time_step** (float): max time step allowed in CVODE.
    delta_t : float
        Evolution time step (sec).
    threshold : float
        A non-positive value. The value belongs to [threshold,0) in label will be set to zero.
    row_in_each_bacth : int
        The row index in the  input batch expected to be compute. 
    advance_type : str
        The type of advancing for GMS (`label` or `evolution`). Type `evolution` can be more tolerant for  calculation error but expect fewer computation cost. 
    

    """
    input_batch = np.load(
        input_batch_path
    )  # load inpu_batch. Too Big batch size  leads to significant disk space cost in sub-process
    dim = input_batch.shape[1]
    try:
        label_batch_row_temp = ctOneStep(input_batch[row_in_each_bacth, :],
                                         gas_config, delta_t, advance_type)
        zero = np.where(label_batch_row_temp < threshold)[0]
        if zero.size == 0:  # negative value exists but >=threshold
            label_batch_row_temp[(label_batch_row_temp < 0)
                                 & (label_batch_row_temp >= threshold
                                    )] = 0  # set value in [threshold,0) to 0.
        else:  # negative value exists and  < threshold
            label_batch_row_temp = np.zeros((1, dim))
    except Exception as e:
        label_batch_row_temp = np.zeros((1, dim))  # evolution failure

    label_batch_row_path = os.path.join(
        cache_batch_path, f'label_batch_adap{row_in_each_bacth}.npy')
    
    if save_data:
        np.save(label_batch_row_path, label_batch_row_temp)
    else:
        return label_batch_row_temp
 

def batchGenLabel(input_path,
                  label_path,
                  gas_config,
                  delta_t,
                  threshold,
                  advance_type,
                  process,
                  para_method='mp',
                  retain_size=False):
    r"""Given the input dataset, use ``multi-processing`` or ``MPI`` to generate the corresponding label dataset.
    
    Parameters
    ---------
    input_path : str
        The input dataset path
        
    label_path : str
        The path to save label dataset.
    gas_config : list or tuple
        Basic configurations for gas reactor. Organized as [fuel, reactor, cantera_max_time_step]. 
        **fuel** (str): fuel species name.  **reactor** (str): reactor type, 'constP' or 'constV'.  **cantera_max_time_step** (float): max time step allowed in CVODE.
    delta_t : float
        Evolution time step (sec).
    threshold : float
        A non-positive value. The value belongs to [threshold,0) in label will be set to zero.
    advance_type : str
        The type of advancing for GMS (`label` or `evolution`). Type `evolution` can be more tolerant for  calculation error but expect fewer computation cost. 
    process : int
        The number of process when using ``multi-processing``.
    para_method : str,optional
        Parallelization method. `mpi` for ``MPI`` and `mp` for ``mutiprocessing``.

    Returns
    -------
    None 
        The label dataset will be saved in **label_path**.

    """
    input = np.load(input_path)
    rows_input, dim_input = input.shape
    t0 = time.time()
    # if para_method == 'mpi':  #assign batches in different cpus
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpu_size = comm.Get_size()
    random_mark = rank
    ## -------Set batch_size and batch_num---------
    func = ceil if cpu_size >= rows_input else floor
    # adaptive batch_size to fullil all the CPUs.
    batch_size = func(rows_input / cpu_size)
    batch_size = max(batch_size, 5000)
    batch_num = ceil(rows_input / batch_size)
    ## -------Set batch_size and batch_num---------
    cache_label_path = f'CacheLabelData{0}'  # main process folder
    batch_index_splits = mpiSplitData(batch_num, rank, cpu_size)
    # if para_method == 'mp':
    #     batch_size = 10000
    #     batch_num = ceil(rows_input / batch_size)
    #     batch_index_splits = range(batch_num)
    #     random_mark = np.random.randint(
    #         10000, 100000)  # random suffix for cache folder
    #     cache_label_path = f'CacheLabelData{random_mark}'

    os.makedirs(cache_label_path, exist_ok=True)

    for batch_index in batch_index_splits:
        ## todo: split input dataset into batches
        cache_batch_path = f'CacheBatchData{random_mark}'
        os.makedirs(cache_batch_path, exist_ok=True)
        time.sleep(1)

        From = batch_index * batch_size
        To = min(rows_input, (batch_index + 1) * batch_size)
        input_batch = input[From:To, :]
        input_batch_path = os.path.join(cache_batch_path,
                                        f'input_batch{batch_index}.npy')
        np.save(input_batch_path, input_batch)

        ## todo: use multi-process to generate corresponding label in each batch
        # p = Pool(process)
        # for row_in_each_bacth in range(
        #         input_batch.shape[0]):  #input_batch.shape[0]
        #     p.apply_async(func=_ctOneStepWrapper,
        #                   args=(
        #                       cache_batch_path,
        #                       input_batch_path,
        #                       batch_index,
        #                       gas_config,
        #                       delta_t,
        #                       threshold,
        #                       row_in_each_bacth,
        #                       advance_type,
        #                   ))

        # p.close()
        # p.join()

        # # todo: concat input and label respectively in each batch. Cantera failure will be excluded when generating label
        # label_batch = bisectionConcat(0, input_batch.shape[0] - 1, dim_input,
        #                               cache_batch_path, 'label_batch_adap')
        
        label_batch = np.zeros_like(input_batch)
        for row_in_each_bacth in range(input_batch.shape[0]):  #input_batch.shape[0]
            label_batch[row_in_each_bacth, :] = _ctOneStepWrapper(cache_batch_path, input_batch_path, batch_index, gas_config, delta_t, threshold, row_in_each_bacth, advance_type, save_data = False )

        input_batch_path = os.path.join(cache_label_path,
                                        f'input_batch{batch_index}.npy')
        label_batch_path = os.path.join(cache_label_path,
                                        f'label_batch{batch_index}.npy')
        np.save(input_batch_path, input_batch)
        np.save(label_batch_path, label_batch) 
        emptyFolder(cache_batch_path, rank)
        # shutil.rmtree(cache_batch_path, ignore_errors=True)
        # time.sleep(2)

    # if para_method=='mpi':
    # print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] rank={rank:^5} finished')

    ## allow_concat
    allow_concat = False
    # if para_method == 'mp':
        # allow_concat = True
    if para_method == 'mpi' and rank == 0 and allowConcatForMPI(
            batch_num, cache_label_path, data_name='label_batch'):
        allow_concat = True

    ## todo: concat all the batches of input and label.
    if allow_concat:
        p1 = time.time()
        input = bisectionConcat(0, batch_num - 1, dim_input, cache_label_path,
                                'input_batch')
        label = bisectionConcat(0, batch_num - 1, dim_input, cache_label_path,
                                'label_batch')
        p2 = time.time()
        print(f"concat time : {p2-p1:.1f} s")
        ## todo: choose samples whose temperature!=0
        if not retain_size:
            print("clear unexpected samples")
            row_pick = np.where(label[:, 0] > 1)[0]
            input = input[row_pick, :]
            label = label[row_pick, :]
            print("input: ", input.shape)
        else:
            print(f"Warning: label dataset might contain zero-value samples since retain_size = {retain_size}")
        np.save(input_path, input)
        np.save(label_path, label)
        
        emptyFolder(cache_label_path, rank)
        # shutil.rmtree(cache_label_path, ignore_errors=True)
        # time.sleep(4)
        t1 = time.time()
        print(f"save and clear time : {t1-p2:.1f} s")
        print(f"input saved in {input_path} with shape {input.shape}")
        print(f"label saved in {label_path} with shape {label.shape}")
        print(f"time step {delta_t}s")
        print(f"batchGenLabel time cost {t1-t0:.2f}s\n")



def minibatchGenLabel(input_path,
                  label_path,
                  gas_config,
                  delta_t,
                  threshold,
                  advance_type,
                  process,
                  retain_size=False):
    r"""Given a mini (size<10w) input dataset, use ``multi-processing`` or ``MPI`` to generate the corresponding label dataset.
        The whole procedure will be completed in single CPU/Process.
    
    Parameters
    ---------
    input_path : str
        The input dataset path
        
    label_path : str
        The path to save label dataset.
    gas_config : list or tuple
        Basic configurations for gas reactor. Organized as [fuel, reactor, cantera_max_time_step]. 
        **fuel** (str): fuel species name.  **reactor** (str): reactor type, 'constP' or 'constV'.  **cantera_max_time_step** (float): max time step allowed in CVODE.
    delta_t : float
        Evolution time step (sec).
    threshold : float
        A non-positive value. The value belongs to [threshold,0) in label will be set to zero.
    advance_type : str
        The type of advancing for GMS (`label` or `evolution`). Type `evolution` can be more tolerant for  calculation error but expect fewer computation cost. 
    process : int
        The number of process when using ``multi-processing``.
    para_method : str,optional
        Parallelization method. `mpi` for ``MPI`` and `mp` for ``mutiprocessing``.

    Returns
    -------
    None 
        The label dataset will be saved in **label_path**.

    """
    input = np.load(input_path)
    rows_input, dim_input = input.shape
    t0 = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpu_size = comm.Get_size()

    batch_size = 10000
    batch_num = ceil(rows_input / batch_size)
    batch_index_splits = range(batch_num)

    cache_label_path = f'CacheLabelData{rank}'  # main process folder
    os.makedirs(cache_label_path, exist_ok=True)

    for batch_index in batch_index_splits:
        ## todo: split input dataset into batches
        cache_batch_path = f'CacheBatchData{rank}'
        os.makedirs(cache_batch_path, exist_ok=True)
        time.sleep(1)

        From = batch_index * batch_size
        To = min(rows_input, (batch_index + 1) * batch_size)
        input_batch = input[From:To, :]
        input_batch_path = os.path.join(cache_batch_path, f'input_batch{batch_index}.npy')
        np.save(input_batch_path, input_batch)

        ## todo: use multi-process to generate corresponding label in each batch
        # p = Pool(process)
        # for row_in_each_bacth in range(input_batch.shape[0]):  #input_batch.shape[0]
        #     p.apply_async(
        #                   func=_ctOneStepWrapper,
        #                   args=(
        #                       cache_batch_path,
        #                       input_batch_path,
        #                       batch_index,
        #                       gas_config,
        #                       delta_t,
        #                       threshold,
        #                       row_in_each_bacth,
        #                       advance_type,
        #                   ))

        # p.close()
        # p.join()

        label_batch = np.zeros_like(input_batch)
        for row_in_each_bacth in range(input_batch.shape[0]):  #input_batch.shape[0]
            label_batch[row_in_each_bacth, :] = _ctOneStepWrapper(cache_batch_path, input_batch_path, batch_index, gas_config, delta_t, threshold, row_in_each_bacth, advance_type, save_data = False )

        ## todo: concat input and label respectively in each batch. Cantera failure will be excluded when generating label
        # label_batch = bisectionConcat(0, input_batch.shape[0] - 1, dim_input,
        #                               cache_batch_path, 'label_batch_adap')
        
        input_batch_path = os.path.join(cache_label_path,
                                        f'input_batch{batch_index}.npy')
        label_batch_path = os.path.join(cache_label_path,
                                        f'label_batch{batch_index}.npy')
        np.save(input_batch_path, input_batch)
        np.save(label_batch_path, label_batch) 
        emptyFolder(cache_batch_path, rank)
        # shutil.rmtree(cache_batch_path, ignore_errors=True)
        # time.sleep(2)

    ## todo: concat all the batches of input and label.
    p1 = time.time()
    input = bisectionConcat(0, batch_num - 1, dim_input, cache_label_path, 'input_batch')
    label = bisectionConcat(0, batch_num - 1, dim_input, cache_label_path, 'label_batch')
    p2 = time.time()
    print(f"concat time : {p2-p1:.1f} s")
    ## todo: choose samples whose temperature!=0
    if not retain_size:
        print("clear unexpected samples")
        row_pick = np.where(label[:, 0] > 1)[0]
        input = input[row_pick, :]
        label = label[row_pick, :]
    else:
        print(f"Warning: label dataset might contain zero-value samples since retain_size = {retain_size}")
    np.save(input_path, input)
    np.save(label_path, label)
    emptyFolder(cache_label_path)
        # shutil.rmtree(cache_label_path, ignore_errors=True)
        # time.sleep(4)
    t1 = time.time()
    print(f"save and clear time : {t1-p2:.1f} s")
    print(f"input saved in {input_path} with shape {input.shape}")
    print(f"label saved in {label_path} with shape {label.shape}")
    print(f"time step {delta_t}s")
    print(f"minibatchGenLabel time cost {t1-t0:.2f}s\n")


    
gas = None