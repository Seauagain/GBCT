"""
Data processing methods.
"""

## system import
import cantera as ct
import numpy as np
from mpi4py import MPI
from math import ceil, floor
import os

## custom import 
from .CanteraTools import ctOneStep, batchGenLabel
from .utils_data import *


class DataProcess():

    def __init__(self) -> None:
        pass


    def initGas(self,mech_path):
        r"""Instantiate cantera.Solution object. 

        Parameters
        ----------
        mech_path : str
            Mechanism input file, could be .yaml, .xml or .cti format.
                    
        """
        self.gas=ct.Solution(mech_path)
    

    @staticmethod
    def preload(dataset):
        r"""Preload the dataset. **dataset** could be `str` or `numpy.ndarray`.
        
        Parameters
        ----------
        dataset : str or numpy.ndarray
            A dataset, could be numpy.ndarray format data or dataset path.
        
        Returns
        -------
        dataset : numpy.ndarray
            The loaded dataset.
            
        """
        if isinstance(dataset, np.ndarray):
            print(f"load data, shape: {dataset.shape}")
            pass
        elif isinstance(dataset,str):
            dataset_path = dataset
            dataset=np.load(dataset)
            print(f"load data from {dataset_path}, shape: {dataset.shape}")
        else:
            raise TypeError(f"expected dataset be <class 'str'> or <class 'np.ndarray'> but got {dataset.__class__}")
        return dataset


    def verifyOneStep(self,input_path, label_path, gas_config, delta_t, test_size=1000):
        r"""Randomly pick some samples in input and label dataset, verify whether they're data pairs.

        Parameters
        ----------
        input_path : str or numpy.ndarray
            The input dataset.
        labe_path : str or numpy.ndarray
            The label dataset.
        gas_config : list or tuple
            Basic configurations for gas reactor. Organized as [fuel, reactor, cantera_max_time_step]. 
            **fuel** (str): fuel species name.  **reactor** (str): reactor type, 'constP' or 'constV'.  **cantera_max_time_step** (float): max time step allowed in CVODE.
        delta_t : float
            Evolution time step (sec).
        test_size : int,optional
            Test size, default 1000.

        """
        input = self.preload(input_path)
        # print(f'load {input_path},shape {input.shape}')
        label = self.preload(label_path)
        # print(f'load {label_path},shape {label.shape}')
        rows = np.random.randint(input.shape[0], size=np.min([test_size, input.shape[0]]))
        max_err = 0
        max_err_row = -1
        for row in rows:
            err = ctOneStep(input[row, :], gas_config, delta_t, advance_type='label') - label[row, :]
            abs_err_max =  np.max(np.abs(err))
            if abs_err_max > max_err:
                max_err = abs_err_max
                max_err_row = row
        if input.shape[0]!=label.shape[0]:
            print(f'Warning: input size {input.shape} differs from label size {label.shape}')
        print(f'randomly verify {test_size} samples with time step {delta_t}s, and max_abs_err={max_err:.2e}, max_err_row={max_err_row}')

    def meanRateFilter(self, manifold_input, manifold_label, input_path, label_path, delta_t, filter_rate, ignore_dims, filter_dims='all', filter_threshold=10, save_data=False):
        r"""Filter input and label dataset by the average changing rate of a given dataset (usually be the manifoid).
        
        Parameters
        ---------
        manifold_input : str or numpy.ndarray
            The manifold input dataset.
        manifold_label : str or numpy.ndarray
            The manifold label dataset.
        input_path : str or numpy.ndarray
            The input dataset.
        label_path : str or numpy.ndarray
            The label dataset.  
        delta_t : float
            Time step between input and label.
        filter_rate : float
            Filteration rate. 
        ignore_dims : list
            Dimensions which will **Not** be filtered. Could be like [0,1,2] or ['P','AR'] etc.
        filter_dims : str,optinal
            Dimensions which will be filtered. Could be 'all','sp',[0,1,2] or ['T','P','OH']. 'all' denotes all dimensions except ignore_dims 
            will be used. 'sp' denotes the species dimensions except ignore_dims will be used.
        save_data : bool,optinal
            Whether save data in input_path and label_path, default False.
  
        """
        manifold_input = self.preload(manifold_input)
        manifold_label = self.preload(manifold_label)
        input = self.preload(input_path)
        label = self.preload(label_path)
        manifold_rate = (manifold_label - manifold_input) / delta_t
        rate = (label - input) / delta_t

        n_row, n_col = rate.shape
        print(f'original dataset shape: {rate.shape}')
        
        # process filter dims and ignore dims
        names = ['T','P'] + self.gas.species_names # ['T','P',species_names]

        if filter_dims=='all':
            filter_dims = range(n_col)
        elif filter_dims=='sp':
            filter_dims = range(2, n_col)

        for i,dim in enumerate(filter_dims):
            if isinstance(dim ,str):
                if dim in ['T','P']:
                    filter_dims[i] = names.index(dim)
                else:
                    filter_dims[i] = self.gas.species_index(dim) + 2
        
        for j,dim in enumerate(ignore_dims):
            if isinstance(dim ,str):
                if dim in ['T','P']:
                    ignore_dims[j] = names.index(dim)
                else:
                    ignore_dims[j] = self.gas.species_index(dim) + 2
        
        ## todo: operate fileration
        rows_del = np.array([], dtype='int')
        for col in filter_dims:
            if col in ignore_dims:
                continue
            else:
                # if np.max(np.abs(manifold_rate[:, col])) > filter_threshold:
                filter_rate_dim = filter_rate
                # else:
                #     filter_rate_dim = 10 # 流形该维度变化率小于10，筛选率默认设置为10倍
                row_max = np.where(rate[:, col] > max(filter_rate_dim * np.max(manifold_rate[:, col]), (1/filter_rate_dim) * np.max(manifold_rate[:, col])))[0]
                rows_del = np.r_[rows_del, row_max]
                row_min = np.where(rate[:, col] < min(filter_rate_dim * np.min(manifold_rate[:, col]), (1/filter_rate_dim) * np.min(manifold_rate[:, col])))[0]
                rows_del = np.r_[rows_del, row_min]


        rows_del = np.unique(rows_del)
        input = np.delete(input, rows_del, axis=0)
        label = np.delete(label, rows_del, axis=0)
        print(f'meanRateFilter will remain data shape {input.shape} since filter_dim={filter_dims} ignore_dims={ignore_dims} filter_rate={filter_rate}')
        if save_data == True:
            np.save(input_path, input)
            np.save(label_path, label)
            print(f'meanRateFilter input saved in {input_path}')
            print(f'meanRateFilter label saved in {label_path}')

    def rangeFilter(self, manifold_input, manifold_label, input_path, label_path, filter_rate, ignore_dims, filter_dims='all', filter_threshold=5e-3, save_data=False):
        r"""
        """
        manifold_input = self.preload(manifold_input)
        manifold_label = self.preload(manifold_label)
        input = self.preload(input_path)
        label = self.preload(label_path)

        n_row,n_col = input.shape
        print(f'original dataset shape: {input.shape}')

        # process filter dims and ignore dims
        names = ['T','P'] + self.gas.species_names # ['T','P',species_names]


        # 判断组分是否存在
        if filter_dims=='all':
            filter_dims = range(n_col)
        elif filter_dims=='sp':
            filter_dims = range(2, n_col)
        for i,dim in enumerate(filter_dims):
            if isinstance(dim ,str):
                if dim in ['T','P']:
                    filter_dims[i] = names.index(dim)
                else:
                    filter_dims[i] = self.gas.species_index(dim) + 2

        for j,dim in enumerate(ignore_dims):
            if isinstance(dim ,str):
                if dim in ['T','P']:
                    ignore_dims[j] = names.index(dim)
                else:
                    ignore_dims[j] = self.gas.species_index(dim) + 2

        ## todo: operate fileration
        rows_del = np.array([], dtype='int')
        for col in filter_dims:
            if col in ignore_dims:
                pass
            else:
                # if np.max(np.abs(manifold_input[:, col])) > filter_threshold:
                #     filter_rate_dim = filter_rate
                # else:
                #     filter_rate_dim = 100

                row_max = np.where(input[:, col] > max(filter_threshold, filter_rate* np.max(manifold_input[:, col])))[0]
                rows_del = np.r_[rows_del, row_max]
        
        rows_del = np.unique(rows_del)
        input = np.delete(input, rows_del, axis=0)
        label = np.delete(label, rows_del, axis=0)
        print(f'rangeFilter will remain data shape {input.shape} since filter_dim={filter_dims} ignore_dims={ignore_dims} filter_rate={filter_rate}')
        if save_data == True:
            np.save(input_path, input)
            np.save(label_path, label)
            print(f'rangeFilter input saved in {input_path}')
            print(f'rangeFilter label saved in {label_path}')

    def TPFilter(self, input_path, label_path, T_range=None, P_range=None, save_data=False):
        r"""Temperature and pressure filteration method. Reserve samples whose temperature or pressure belongs to the given range.
        
        Parameters
        ----------
        input_path : str or numpy.ndarray
            The input dataset.
        label_path : str or numpy.ndarray
            The label dataset.  
        T_range : list or tuple,optional
            Temperature range. Default None.
        P_range : list or tuple,optional
            Pressure range. Default None.
        save_data : bool,optinal
            Whether save data in input_path and label_path, default False.

 
        """
        input = self.preload(input_path)
        label = self.preload(label_path)
        ## T filteration
        if T_range:
            rows_pick = np.where((input[:,0] > T_range[0])&(input[:,0] < T_range[1]))[0]
            input = input[rows_pick,:]
            label = label[rows_pick,:]
            print(f'TPFilter will remain data shape {input.shape} since T_range={T_range}')
        ## P(atm) filteration
        if P_range:
            rows_pick = np.where((input[:,1] > P_range[0])&(input[:,1] < P_range[1]))[0]
            input = input[rows_pick,:]
            label = label[rows_pick,:]
            print(f'TPFilter will remain data shape {input.shape} since P_range={P_range}')

        
        if save_data == True:
            # print(np.min(input))
            np.save(input_path, input)
            np.save(label_path, label)
            print(f'TPFilter input saved in {input_path}')
            print(f'TPFilter label saved in {label_path}')

    def splitManifold(self, input_path, diff_tol=30):
        """
        区分数据集中稳态流形和火焰流形
        """
        t0 = time.time()
        input = self.preload(input_path)
        sample_num, dim = input.shape
        # label = self.preload(label_path)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        cpu_size = comm.Get_size()
        para_method = "mpi"

        ## -------Set batch_size and batch_num---------
        func = ceil if cpu_size >= sample_num else floor
        # adaptive batch_size to fullil all the CPUs.
        batch_size = func(sample_num / cpu_size)
        batch_num = ceil(sample_num / batch_size)
        ## -------Set batch_size and batch_num---------
        batch_index_splits = mpiSplitData(batch_num, rank, cpu_size) # assign batches to all the CPUs
        cache_path = f'CacheIgnition{0}'
        os.makedirs(cache_path, exist_ok=True)

        for batch_index in batch_index_splits:
            ign_row = []
            From = batch_index * batch_size
            To = min(sample_num, (batch_index + 1) * batch_size)
            # input_batch = input[From:To, :]
            for i in range(From, To):
                # if os.path.exists(f"CacheIgnition0/ignition_row{i}.npy"):
                    # os.system(f"rm CacheIgnition0/ignition_row{i}.npy")
                state = input[i, :]
                init_temperature = state[0] #初始温度

                self.gas.TPY = state[0], state[1] * ct.one_atm, state[2:]
                try:
                    self.gas.equilibrate('HP',solver="gibbs",rtol=1e-1,max_steps=20000,max_iter=20000)
                    equi_temperature = self.gas.T # 平衡温度
                    diff_temperature = np.abs(equi_temperature - init_temperature)
                    if diff_temperature > diff_tol:
                        ign_row.append(i)
                except:
                    pass
            ign_row_path = os.path.join(cache_path, f"ign_batch{batch_index}.npy")
            np.save(ign_row_path, np.array(ign_row).reshape(-1,1))
            print(f"saved in {ign_row_path}, shape : {len(ign_row)}")
        
        allow_concat = False
        if  rank == 0 and allowConcatForMPI(
            batch_num, cache_path, data_name='ign_batch'):
            allow_concat = True

        if allow_concat:
            rows = bisectionConcat(0, batch_num - 1, 1,
                                   cache_path, 'ign_batch')
            
            index_path = f"{os.path.splitext(input_path)[0]}_index.npy"
            np.save(index_path, rows)
            # shutil.rmtree(cache_path, ignore_errors=True)
            emptyFolder(cache_path)
            t1 = time.time()
            print(f'Ignition manifold shape : {rows.shape}, splitMF time cost: {t1-t0:.2f}s')



