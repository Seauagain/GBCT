
## system import 
import numpy as np
import os
import cantera as ct
import shutil, time
import sys
import logging
from math import ceil, floor
from multiprocessing import Pool
try:
    from mpi4py import MPI
except:
    pass

## custom import 
from .CanteraTools import ctOneStep, batchGenLabel
from .DataProcess import DataProcess
from .utils_data import *



class SampleMethod():

    def __init__(self) -> None:
        pass

    def initGas(self, mech_path):
        r"""Instantiate cantera.Solution object. 

        Parameters
        ----------
        mech_path : str
            Mechanism input file, could be .yaml, .xml or .cti format.
                    
        """
        self.gas = ct.Solution(mech_path)
    
    def initGenX(self, fuel, num, xdir, saveData=False):
        r"""tiny-scale samples druing the pre-ignition process."""
        index_O2 = self.gas.species_index('O2') + 2
        index_fuel = self.gas.species_index(fuel) + 2
        X = np.zeros((num, 2 + self.gas.n_species))  #TPY
        for dim in range(2, 2 + self.gas.n_species):  #从组分维度开始
            if dim in [index_fuel, index_O2]:
                X[:, dim] = self.logEachDimension(num, low_degree=-5, up_degree=0)
            else:
                zeroRate = 0.8
                rows1 = int(zeroRate * num)
                X[0:rows1, dim] = -np.inf
                # X[rows1:, dim] = X[:, dim] * ((-12) - (-25)) + (-25)
                X[rows1:,
                  dim] = self.multiLogEachDimension(num - rows1, -25, -12)

        X[:, 0] = np.random.uniform(500, 3600, num)  #温度
        X[:, 1] = np.random.uniform(0.5, 2, num)  #压强
        X[:, 2:] = self.exp10(X[:, 2:])
        index_N2 = self.gas.species_index('N2') + 2
        X[:, index_N2] = np.random.uniform(0.2, 0.9, num)
        try:
            index_Ar = self.gas.species_index('Ar') + 2
            X[:, index_Ar] = 0  #Ar=0，有些机理可能没Ar
        except:
            pass
        X[:, 2:] = self.normalization(X[:, 2:])
        if saveData == True:
            np.save(xdir, X)
            print(f'initGenX save data: {X.shape}')

    def multiScaleGenInput(self,
                           input_path,
                           sample_num,
                           fuel,
                           degree_range,
                           T_range,
                           P_range,
                           each_dim,
                        sample_offset,
                           inert_gas=[],
                           save_data=False):
        r"""Use multi-scale method to generate an input dataset organized as :math:`T,P(atm),Y`. The chemical dataset will be saved in **input_path**.
        
        Parameters
        ---------
        input_path : str 
            Path to save input dataset.
        sample_num : int
            Total size of the dataset.
        fuel : str
            Fuel name.
        degree_range : numpy.ndarray
            Log degree lower and upper bound.
        T_range : list or tuple
            Range of temperature(K) e.g. [1000,3000].
        P_range : list or tuple
            Range of pressure(atm) e.g. [0.5,2].
        each_dim : str
            Method to obtain degree distribution for each species dimension i.e. 'log' , 'mlog' or 'tmlog'. 'log' means LogEachDimension and 'mlog' means multiLogEachDimension.
        inert_gas : list,optional
            Inert gases whose mass fraciton should be zero. e.g ['Ar','He']. Default [].
        save_data : bool
            Whether save input dataset, default False.
       
            
        """
        print('using multiScaleGenInput')
        input = np.random.rand(sample_num, 2 + self.gas.n_species)  #TPY
        each_dim_types = {
            'log': self.logEachDimension,
            'mlog': self.multiLogEachDimension,
            'tmlog': self.twoMultiLogEachDimension
        }
        dim_func = each_dim_types[each_dim]

        for dim in range(2, 2 + self.gas.n_species):
            input[:, dim] = dim_func(sample_num, degree_range[0, dim],
                                     degree_range[1, dim])

        input[:, 0] = np.random.uniform(T_range[0], T_range[1],
                                        sample_num)  # temperature
        input[:, 1] = np.random.uniform(P_range[0], P_range[1],
                                        sample_num)  # pressure

        ## todo: deal with O2/fuel/N2
        # index_O2 = self.gas.species_index('O2') + 2
        index_fuel = self.gas.species_index(fuel) + 2
        input[:, 2:] = self.exp10(input[:, 2:])
        index_N2 = self.gas.species_index('N2') + 2
        input[:, index_N2] = np.random.uniform(0.6, 1, sample_num)

        ## todo: deal with inert gases whose mass fraciton should be zero
        for inert in inert_gas:
            try:
                idx = self.gas.species_index(inert) + 2
                input[:, idx] = 0
            except:
                pass
        input = self.adjustBythreshold(input, sample_offset)
        input[:, 2:] = self.normalization(input[:, 2:])
        print(f'multiScaleGenInput ({each_dim}) generates data with shape {input.shape}')
        if save_data == True:
            np.save(input_path, input)
            print(f'multiScaleGenInput data saved in {input_path}')

    @staticmethod
    def normalization(data):
        r"""Normalization by summation of axis=1."""
        return data / np.sum(data, axis=1, keepdims=True)

    @staticmethod
    def adjustBythreshold(input, sample_threshold=1e-25):
        r"""Set values<= **sample_threshold** in input to zero.

        Parameters
        ----------
        input : numpy.ndarray
            The input dataset
        sample_threshold : float,optional
            The threhold when sampling.
        
        Returns
        ------
        input : numpy.ndarray
            The processed data.

        """
        input[:, 2:] -= sample_threshold
        input[input < 0] = 0
        return input

    @staticmethod
    def exp10(degree):
        return 10**degree

    @staticmethod
    def logDegreeBound(input, sample_threshold=1e-25):
        r"""Given a non-negative chemical dataset [T,P(atm),Y], return the approximate log degree bound of species mass fraction.
            
        Parameters
        ----------
        input : numpy.ndarray
            The chemical dataset. Usually be the manifold.
        sample_threshold : float,optional
            Avoid log singularity when encountering zero values, i.e. log(sample_threshold + 0). Default 1e-25.
        
        Returns
        -------
        log_degree_bound : numpy.ndarray
            Lower and upper bound under log scale of mass fraction. axis=0-->lower bound; axis=1-->upper bound
                                            
         """
        _, dims = input.shape
        log_degree_bound = np.zeros((2, dims), dtype=int)
        for dim in range(2, dims):
            log_degree_bound[0, dim] = floor(-2 + np.min(
                np.log10(sample_threshold + input[:, dim])))  # lower bound
            log_degree_bound[1, dim] = min(
                0, 0 + ceil(np.max(np.log10(sample_threshold +
                                            input[:, dim]))))  # upper bound
        return log_degree_bound

    @staticmethod
    def logEachDimension(sample_num, low_degree, up_degree):
        r"""Uniform distribution under log scale."""
        degree_distribution = np.random.uniform(low_degree,
                                                up_degree,
                                                size=sample_num)
        # print(f'\rcall logEachDimension: {degree_distribution.shape}',end='\t')
        return degree_distribution

    @staticmethod
    def twoMultiLogEachDimension(sample_num, low_degree, up_degree):
        if up_degree - low_degree <= 1:
            return self.multiLogEachDimension(sample_num, low_degree,
                                              up_degree)
        mid_degree = low_degree + (up_degree - low_degree) // 2
        batch_num_up = up_degree - mid_degree
        batch_num_low = mid_degree - low_degree
        batch_size = ceil(sample_num / (batch_num_up + batch_num_low))
        degree_distribution = np.array([]).reshape(0, )
        for index in range(low_degree, mid_degree):
            From = index * batch_size
            To = (index + 1) * batch_size
            sub_sample_num = To - From
            temp = np.random.uniform(low_degree,
                                     index + 1,
                                     size=sub_sample_num)
            # temp = np.random.rand(samplesNum, 1) * (up - index) + index
            degree_distribution = np.r_[degree_distribution, temp]
        rest_num = sample_num - degree_distribution.shape[0]
        for index in range(mid_degree - 1, up_degree):
            From = index * batch_size
            To = min(rest_num, (index + 1) * batch_size)
            sub_sample_num = To - From
            temp = np.random.uniform(index, up_degree, size=sub_sample_num)
            # temp = np.random.rand(samplesNum, 1) * (up - index) + index
            degree_distribution = np.r_[degree_distribution, temp]
        permutation = np.random.permutation(sample_num)
        degree_distribution = degree_distribution[permutation]
        # print('\rtwoLogEachDimension:', X.shape, end='')
        return degree_distribution

    @staticmethod
    def multiLogEachDimension(sample_num, low_degree, up_degree):
        r"""Multiple uniform distribution under different log scales."""
        batch_num = up_degree - low_degree
        batch_size = ceil(sample_num / batch_num)
        degree_distribution = np.array([]).reshape(0, )
        for index in range(low_degree, up_degree):
            From = index * batch_size
            To = min(sample_num, (index + 1) * batch_size)
            sub_sample_num = To - From
            temp = np.random.uniform(index, up_degree, size=sub_sample_num)
            degree_distribution = np.r_[degree_distribution, temp]
        permutation = np.random.permutation(sample_num)  # must shuffle
        degree_distribution = degree_distribution[permutation]
        # print(f'\rcall multiLogEachDimension: {degree_distribution.shape}', end='\t')
        return degree_distribution

    def genInitState(self, case_num, fuel, oxidizer, param_ranges):
        r"""Given T,P(atm),Phi return the corresponding state vectors T,P(atm),Y.

        Parameters
        ----------
        case_num : int
            Total number of cases.
        fuel : str
            Fuel name.
        oxidizer : str 
            Oxidizer, usually 'O2:1.0,N2:3.76'.
        param_ranges :dict
            The initial range of T, P, Phi.

        Returns
        -------
        init_state : numpy.ndarray
            State vectors.
        
        """
        init_cond = np.random.rand(case_num, 3)
        init_cond[:, 0] = np.random.uniform(param_ranges['Phi'][0], param_ranges['Phi'][1],
                                            case_num)
        init_cond[:, 1] = np.random.uniform(param_ranges['T'][0], param_ranges['T'][1],
                                            case_num)
        init_cond[:, 2] = np.random.uniform(param_ranges['P'][0], param_ranges['P'][1],
                                            case_num)

        init_state = np.zeros((case_num, 2 + self.gas.n_species))
        for row in range(case_num):
            gas.set_equivalence_ratio(init_cond[row, 0], fuel, oxidizer)
            gas.TP = init_cond[row, 1], init_cond[row, 2]
            state = np.hstack(gas.TPY).reshape(1, -1)
            init_state[row, :] = state.copy()
        return init_state

    
    def genInitMixture(self, sample_num, main_species=["O2"], temperature_range=[300, 800], pressure_range=[0.5, 1.5], N2_range=[0.55, 0.8]):
        """初始物质未必按照当量比给定，eg. sandia. 给定初始场主要物质的浓度范围，生成数据"""
        data = np.zeros((sample_num, 2 + gas.n_species))  #TPY

        data[:, 0] = np.random.uniform(temperature_range[0], temperature_range[1], sample_num)
        data[:, 1] = np.random.uniform(pressure_range[0], pressure_range[1], sample_num)
        
        for main_specie in main_species:
            index_sp = 2 + gas.species_names.index(main_specie)
            data[:, index_sp] = np.random.uniform(0.0, 1, sample_num)
        
        index_N2 = gas.species_index('N2') + 2
        data[:, index_N2] = np.random.uniform(N2_range[0], N2_range[1], sample_num)
        
        species_index = [i for i in range(2, 2 + gas.n_species) if i != index_N2]
        data[:, species_index] = data[:, species_index] / np.sum(data[:, species_index], axis=1, keepdims=True) * (1 - data[:, index_N2].reshape(-1, 1))
        return data





# def TpPhi2state(self, init_cond, state):
#     """"init_cond = (num, 3)"""

def _zeroDimAdapSampleWrapper(input_batch_path, cache_batch_path,
                              row_in_each_batch, gas_config, fuel_index,
                              threshold, delta_t, sample_time_resolve,
                              gradT_threshold, samplerate_high_gradT,
                              samplerate_low_gradT, max_march_time):
    r"""Zero-dimensional adaptive manifold sampling method. This function is a runner for multi-processing.

    Parameters
    ----------
    input_batch_path : str
        The input batch path.
    cache_batch_path : str
        Where temporary file will be saved in.
    row_in_each_batch : int
        The index in input batch which will be taken as an initial state for evolution. Range from 0 to `input_batch.size`-1.
    gas_config : list or tuple
        Basic configurations for gas reactor. Organized as [fuel, reactor, cantera_max_time_step]. 
        **fuel** (str): fuel species name.  **reactor** (str): reactor type, 'constP' or 'constV'.  **cantera_max_time_step** (float): max time step allowed in CVODE.
    fuel_index : int
        The fuel index in species names. Could be obtained by gas.species_index(fuel).
    threshold : float
        A non-positive value. The values in [threshold, 0) will be set to zero when generating labels.
    delta_t : float
        Time step for generating labels.
    sample_time_resolve : float
        Time step between adjacent sampling points (less than **delta_t**). Note: not all the sampling points will be reserved.
    gradT_threshold : float
        Threshold for temporal changing rate of temperature.
    samplerate_high_gradT : float
        Sampling rate for high temperature changing rate.
    samplerate_low_gradT : float
        Sampling rate for low temperature changing rate.
    max_march_time : float
        Maximum allowable evolution time. 

    Returns
    -------
    
    """
    input_batch = np.load(input_batch_path)
    dim = input_batch.shape[1]
    state = input_batch[row_in_each_batch, :]
    # state = init_state[index_state, :]
    manifold = state.reshape(1, -1)
    # Ratio = int(delta_t / Sample_time_Resolve)  #预测步长/
    sample_iter = 0
    state_current = manifold[0, :].reshape(1, -1)
    ignition = 0  # ignition time
    max_heatRR = 0  # current max gradient of temperature
    current_fuel = state_current[0, fuel_index + 2]  #当前燃料的摩尔分数
    final_fuel = state_current[0, fuel_index + 2] * 1e-3  #设定最终的燃料摩尔分数

    while True:
        try:
            state_next = ctOneStep(state=state_current,
                                   gas_config=gas_config,
                                   delta_t=sample_time_resolve,
                                   advance_type='label')
            # print("sample_iter: ", sample_iter)
            zero = np.where(state_next < threshold)[0]
            if zero.size == 0:  # negative value exists but >=threshold
                state_next[(state_next < 0)
                           & (state_next >= threshold
                              )] = 0  # set value in [threshold,0) to 0.
            else:
                break
        except Exception as e:
            logging.info(f'{input_batch_path} error {e}')
            print(e)
            break
        sample_iter = sample_iter + 1
        heatRR = np.abs(state_next[0, 0] -
                        state_current[0, 0]) / sample_time_resolve
        current_fuel = state_next[0, fuel_index + 2]
        # 大热释放率*概率被选中
        if heatRR > gradT_threshold:
            if np.random.rand(1) < samplerate_high_gradT:
                manifold = np.r_[manifold, state_next.copy()]
        #  小热释放率*概率被选中
        else:
            if np.random.rand(1) < samplerate_low_gradT:
                manifold = np.r_[manifold, state_next.copy()]

        #计算点火延迟，更新最大热释放率
        march_time = sample_iter * sample_time_resolve * 1e3  #推进时间，单位：ms
        if heatRR > max_heatRR:  # 计算点火延迟
            ignition = march_time
            max_heatRR = heatRR

        # print(f"sample_iter: {sample_iter} march_time: {march_time:.4f} max_march_time: {max_march_time}")

        ## break condition
        temperature_tol = 1e-3
        if (np.abs(state_next[0, 0] - state_current[0, 0]) < temperature_tol and
                current_fuel < final_fuel) or (march_time > max_march_time):
            break
        state_current = state_next

    # print('manifold_batch_adap.size', manifold.shape)
    manifold_temp_path = os.path.join(
        cache_batch_path, f'manifold_batch_adap{row_in_each_batch}.npy')
    np.save(manifold_temp_path, manifold)


def zeroDimManifoldSample(init_state,
                          gas_config,
                          threshold,
                          delta_t,
                          sample_time_resolve,
                          gradT_threshold,
                          samplerate_high_gradT,
                          samplerate_low_gradT,
                          max_march_time,
                          manifold_path,
                          process,
                          para_method='mp'):
    r"""
    Main funtion entrance for zero-dimensional adaptive manifold sampling method.

    Parameters
    ----------
    init_state : numpy.array
        Initial states (`T,P(atm),Y`).
    gas_config : list or tuple
        Basic configurations for gas reactor. Organized as [fuel, reactor, cantera_max_time_step]. 
        **fuel** (str): fuel species name.  **reactor** (str): reactor type, 'constP' or 'constV'.  **cantera_max_time_step** (float): max time step allowed in CVODE.
    threshold : float
        A non-positive value. The values in [threshold, 0) will be set to zero when generating labels.
    delta_t :float
        Time step for generating labels.
    sample_time_resolve : float
        Time step between adjacent sampling points (less than **delta_t**). Note: not all the sampling points will be reserved.
    gradT_threshold : float
        Threshold for temporal changing rate of temperature.
    samplerate_high_gradT : float
        Sampling rate for high temperature changing rate.
    samplerate_low_gradT : float
        Sampling rate for low temperature changing rate.
    max_march_time : float
        Maximum allowable evolution time. 
    manifold_path : str
        The path to save manifold.
    process : int
        Process number for ``multi-processing``.
    para_method : str,optional
        Parallelization method. `'mp'` for ``multi-processing`` and `'mpi'` for ``MPI``.

    Returns
    -------

    """
    sample_num, dim_input = init_state.shape
    t0 = time.time()
    if para_method == 'mpi':
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        cpu_size = comm.Get_size()
        random_mark = rank
        ## -------Set batch_size and batch_num---------
        func = ceil if cpu_size >= sample_num else floor
        # adaptive batch_size to fullil all the CPUs.
        batch_size = func(sample_num / cpu_size)
        batch_num = ceil(sample_num / batch_size)
        ## -------Set batch_size and batch_num---------
        cache_manifold_path = f'CacheManifold{0}'
        batch_index_splits = mpiSplitData(batch_num, rank, cpu_size)
    if para_method == 'mp':
        batch_size = 10
        batch_num = ceil(sample_num / batch_size)
        batch_index_splits = range(batch_num)
        random_mark = np.random.randint(
            10000, 100000)  # random suffix for cache folder
        cache_manifold_path = f'CacheManifold{random_mark}'

    os.makedirs(cache_manifold_path, exist_ok=True)
    fuel_index = gas.species_index(gas_config[0])

    for batch_index in batch_index_splits:
        ## todo: split input dataset into batches
        cache_batch_path = f'CacheManifoldBatch{random_mark}'
        os.makedirs(cache_batch_path, exist_ok=True)
        time.sleep(1)

        From = batch_index * batch_size
        To = min(sample_num, (batch_index + 1) * batch_size)
        input_batch = init_state[From:To, :]
        input_batch_path = os.path.join(cache_batch_path,
                                        f'input_batch{batch_index}.npy')
        np.save(input_batch_path, input_batch)

        p = Pool(process)
        for row_in_each_batch in range(
                input_batch.shape[0]):  #input_batch.shape[0]
            p.apply_async(func=_zeroDimAdapSampleWrapper,
                          args=(
                              input_batch_path,
                              cache_batch_path,
                              row_in_each_batch,
                              gas_config,
                              fuel_index,
                              threshold,
                              delta_t,
                              sample_time_resolve,
                              gradT_threshold,
                              samplerate_high_gradT,
                              samplerate_low_gradT,
                              max_march_time,
                          ))

        p.close()
        p.join()

        # print("x.get: ", x.get())

        ## todo: concat input and label respectively in each batch. Cantera failure will be excluded when generating label
        manifold_batch = bisectionConcat(0, input_batch.shape[0] - 1,
                                         dim_input, cache_batch_path,
                                         'manifold_batch_adap')
        # shutil.rmtree(cache_batch_path, ignore_errors=True)
        # print(f"zerod rank:{rank:^5}")
        emptyFolder(cache_batch_path)
        manifold_batch_path = os.path.join(cache_manifold_path,
                                           f'manifold_batch{batch_index}.npy')
        np.save(manifold_batch_path, manifold_batch)
        time.sleep(4)

    allow_concat = False
    if para_method == 'mp':
        allow_concat = True
    if para_method == 'mpi' and rank == 0 and allowConcatForMPI(
            batch_num, cache_manifold_path, data_name='manifold_batch'):
        allow_concat = True

    if allow_concat:
        manifold = bisectionConcat(0, batch_num - 1, dim_input,
                                   cache_manifold_path, 'manifold_batch')
        # shutil.rmtree(cache_manifold_path, ignore_errors=True)
        np.save(manifold_path, manifold)
        emptyFolder(cache_manifold_path)
        t1 = time.time()
        print(
            f'manifold sampling size: {manifold.shape}, zeroDimManifoldSample time cost {t1-t0:.2f}s\n'
        )


def _oneDimSampleWrapper(
    temperature,
    pressure, 
    equivalence_ratio, 
    fuel, 
    oxidizer, 
    cache_flame_path,
    width,
    loglevel,
    HRR_threshold,
    lowHRR_sample_rate,
    highHRR_sample_rate):
    r"""One-dimensional flame sampling method. This is a runner for multi-processing.
    More details see cantera.FreeFlame_ .  

    .. _cantera.FreeFlame: https://cantera.org/documentation/docs-2.5/sphinx/html/cython/onedim.html#freeflame

    Parameters
    ----------
    temperature : float
        Initial temperature of premixed gas.
    pressure : float, 
        xxxxx
    Phi : float
        Equivalence_ratio.
    fuel : str
        Fuel name.
    oxidizer : str
        Oxidizer name.
    cache_flame_path : str
        Path for temporary .npy  files.
    width : float
        Defines a grid on the interval [0, width] with internal points determined automatically by the solver.
    loglevel : int
        Integer flag controlling the amount of diagnostic output. Zero suppresses all output, and 5 produces very verbose output.
    HRR_threshold : 
        Threshold for temporal changing rate of temperature.
    lowHRR_sample_rate : float
        Sampling rate for low temperature changing rate.

    Returns
    ------
    
    """
    print(f'Initial temperature={temperature:.2f} K, p={pressure:.2f} atm, Phi={equivalence_ratio:.2f}')
    gas.TP = temperature, ct.one_atm * pressure
    gas.set_equivalence_ratio(equivalence_ratio, fuel, oxidizer)
    # temperature0 = float(init_state[0])
    # P0 = float(init_state[1])

    # print(init_state)
    # gas.TPY = init_state[0], ct.one_atm * init_state[1], init_state[2:]

    initial_grid = np.linspace(0, width, 800)
    f = ct.FreeFlame(gas, initial_grid)
    # f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    f.transport_model = 'Mix'


    try:
        # import eventlet#导入eventlet这个模块
        # eventlet.monkey_patch()#必须加这条代码
        # with eventlet.Timeout(600, False):#设置超时时间为2秒
        f.solve(loglevel=loglevel, auto=True)
        mass_fraction = f.Y
        T = f.T.reshape(-1, 1)
        P = np.ones((T.shape[0], 1)) * f.P / ct.one_atm
        state = np.c_[T, P, mass_fraction.T]
        # print('state shape', state.shape)
        HRR = f.heat_release_rate
        # print("HRR.shape: ", HRR.shape)
        # print("state.shape: ", state.shape)
        # Data = np.r_[Data, state.reshape(-1, 2 + gas.n_species)]
        rows_pick = []
        for row in range(state.shape[0]):
            prob = np.random.rand(1)
            if HRR[row] > HRR_threshold:
                if prob < highHRR_sample_rate:
                    rows_pick.append(row)
            elif prob < lowHRR_sample_rate:
                rows_pick.append(row)
            # rows_pick.append(row)
        state = np.abs(state)
        state[:, 2:] = state[:, 2:] / np.sum(state[:, 2:], axis=1, keepdims=True)
        temp_flame = state[rows_pick, :]
        if temp_flame.shape[0]==0:
            temp_flame = init_state.copy().reshape(1, -1)
    except Exception as e:
        # print(e)
        temp_flame = init_state.copy().reshape(1, -1)

    # else:
    #     signal.alarm(0)
    # state[:, 2:] = self.normalization(X[:, 2:])
    # temp_flame_path = os.path.join(cache_flame_path, f'input_flame{temperature}.npy')
    np.save(cache_flame_path, temp_flame)
    print(f'Initial temperature={temperature:.2f} K, p={pressure:.2f} atm, flame state shape={temp_flame.shape}')


def oneDimFlameSampleMPI(flame_save_path,
                         working_conds,
                         sample_num,
                         fuel,
                         oxidizer,
                         width,
                         loglevel,
                         HRR_threshold,
                         lowHRR_sample_rate,
                         highHRR_sample_rate):
    """Use `MPI` for one-dimensional flame sampling.

    """
    t0 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpu_size = comm.Get_size()
    random_mark = 1  # random suffix for cache folder
    ## -------Set batch_size and batch_num---------
    func = ceil if cpu_size >= sample_num else floor
    # adaptive batch_size to fullil all the CPUs.
    # batch_size = func(sample_num / cpu_size)
    batch_size = 1
    batch_num = ceil(sample_num / batch_size)
    ## -------Set batch_size and batch_num---------
    cache_manifold_path = f'CacheFlame{random_mark}'
    batch_index_splits = mpiSplitData(batch_num, rank, cpu_size)
    os.makedirs(cache_manifold_path, exist_ok=True)
    for batch_index in batch_index_splits:
        ## todo: split input dataset into batches
        init_cond = working_conds[batch_index, :]
        T, P, Phi = init_cond.tolist()
        # print(T,P,Phi)
        # exit(0)
        # state = states[batch_index, :] # single state, one-dimensional vector (input_dim,)
        input_batch_path = os.path.join(cache_manifold_path, f'input_flame{batch_index}.npy')
        _oneDimSampleWrapper(T, P, Phi, fuel, oxidizer, input_batch_path, width,
                             loglevel, HRR_threshold, lowHRR_sample_rate, highHRR_sample_rate)

    allow_concat = False
    if rank == 0 and allowConcatForMPI(
            batch_num, cache_manifold_path, data_name='input_flame'):
        allow_concat = True
    
    dim_input = gas.n_species + 2
    if allow_concat:
        flame = bisectionConcat(0, batch_num - 1, dim_input, cache_manifold_path, data_name='input_flame')
        # shutil.rmtree(cache_manifold_path, ignore_errors=True)
        np.save(flame_save_path, flame)
        emptyFolder(cache_manifold_path)
        t1 = time.time()
        print(f'oneD flame sampling size: {flame.shape}, oneDimFlameSampleMPI time cost {t1-t0:.2f}s\n')



def _contouterFlowSampleWrapper(
    T0,
    p0,
    fuel, 
    oxidizer,
    cache_flame_path):
    
    width = 0.04
    loglevel = 0
    gas.TP = gas.T, p0 * ct.one_atm
    initial_grid = np.linspace(0, width, 100)
    f = ct.CounterflowDiffusionFlame(gas, initial_grid)
    p = ct.one_atm  # pressure
    tin_f = T0  # fuel inlet temperature
    tin_o = T0  # oxidizer inlet temperature
    mdot_o = 0.72  # kg/m^2/s
    mdot_f = 0.34  # kg/m^2/s
    # Set the state of the two inlets
    f.fuel_inlet.mdot = mdot_f
    f.fuel_inlet.X = oxidizer
    f.fuel_inlet.T = tin_f

    f.oxidizer_inlet.mdot = mdot_o
    f.oxidizer_inlet.X = fuel
    f.oxidizer_inlet.T = tin_o

    f.boundary_emissivities = 0.0, 0.0
    f.radiation_enabled = False
    f.set_refine_criteria(ratio=4, slope=0.2, curve=0.3, prune=0.04)
    try: 
        f.solve(loglevel, auto=True)
        mass_fraction = f.Y
        T = f.T.reshape(-1, 1)
        P = np.ones((T.shape[0], 1)) * f.P / ct.one_atm
        state = np.c_[T, P, mass_fraction.T]
        # print('state shape', state.shape)
        HRR = f.heat_release_rate
        state = np.abs(state)
        state[:, 2:] = state[:, 2:] / np.sum(state[:, 2:], axis=1, keepdims=True)
    except:
        state = np.zeros((1, gas,n_species+2))

    np.save(cache_flame_path, state)
    print(f'Temperature={T0:.2f} K, P={p0:.2f} atm, flame state shape={state.shape}')



def counterFlameSampleMPI(flame_path,
                         working_conds,
                         sample_num,
                         fuel,
                         oxidizer):

    t0 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpu_size = comm.Get_size()
    random_mark = 1  # random suffix for cache folder
    ## -------Set batch_size and batch_num---------
    func = ceil if cpu_size >= sample_num else floor
    # adaptive batch_size to fullil all the CPUs.
    # batch_size = func(sample_num / cpu_size)
    batch_size = 1
    batch_num = ceil(sample_num / batch_size)
    ## -------Set batch_size and batch_num---------
    cache_manifold_path = f'CacheCounterFlame{random_mark}'
    batch_index_splits = mpiSplitData(batch_num, rank, cpu_size)
    os.makedirs(cache_manifold_path, exist_ok=True)
    for batch_index in batch_index_splits:
        ## todo: split input dataset into batches
        init_cond = working_conds[batch_index, :]
        T, p, = init_cond.tolist()
        # state = states[batch_index, :] # single state, one-dimensional vector (input_dim,)
        cache_flame_path = os.path.join(cache_manifold_path, f'input_flame{batch_index}.npy')
        _contouterFlowSampleWrapper( T, p, fuel, oxidizer, cache_flame_path)

    allow_concat = False
    if rank == 0 and allowConcatForMPI(batch_num, cache_manifold_path, data_name='input_flame'):
        allow_concat = True
    
    dim_input = gas.n_species + 2
    if allow_concat:
        flame = bisectionConcat(0, batch_num - 1, dim_input, cache_manifold_path, data_name='input_flame')
        # shutil.rmtree(cache_manifold_path, ignore_errors=True)
        np.save(flame_path, flame)
        emptyFolder(cache_manifold_path)
        t1 = time.time()
        print(f'counter flame sampling size: {flame.shape}, counterFlameSampleMPI time cost {t1-t0:.2f}s\n')


gas = None
