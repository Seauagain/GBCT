"""Visualization"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import cantera as ct
import os
import re
from math import ceil, floor
from distutils.spawn import find_executable

matplotlib.use('AGG')
import warnings

warnings.filterwarnings('ignore')


## display phase diagram and distribution histogram of chemical dataset
class VisualArt():

    def __init__(self) -> None:
        pass

    def initGas(self, mech_path):
        r""" instantiate cantera.Solution object 

        Parameters
        ----------
        mech_path : str
            mechanism input file, could be .yaml, .xml or .cti format.

        Returns
        -------
                
        """
        self.gas = ct.Solution(mech_path)

    def phaseDiagram(self,
                     input,
                     label,
                     delta_t,
                     data_name,
                     show_temperature,
                     size_show=500000,
                     dpi=200):
        r"""Draw and save the phase diagram of a chemical dataset (TPY).

        Parameters
        ----------
        input : str or numpy.ndarray
            Input dataset.
        label : str or numpy.ndarray
            Label dataset.
        delta_t : float
            Time step between input and label.
        data_name : str
            The name of dataset, will be used when saving figures.
        size_show : int, optional
            Random pick samples in dataset to show. Default 500,000.
        dpi : int, optional
            The dpi used to save figure. Default 200.
        
        Returns
        -------

                
        """
        ## check input class type
        if isinstance(input, np.ndarray):
            pass
        elif isinstance(input, str):
            input = np.load(input)
            label = np.load(label)
        else:
            raise TypeError(
                f"expected input be <class 'str'> or <class 'numpy.ndarray'> but got {input.__class__}"
            )
        
        ## choose data by temperature
        rows = np.where((input[:, 0] >= show_temperature[0])
                        & (input[:, 0] <= show_temperature[1]))[0]
        input = input[rows, :]
        label = label[rows, :]

        ## shuffle datatset and randomly pick data
        permutation = np.random.permutation(input.shape[0])
        input = input[permutation]
        label = label[permutation]
        input = input[:size_show, :]
        label = label[:size_show, :]
        print(f'phase diagram takes data size {input.shape}')
        print(f"the display range of temperature is {show_temperature}K")

        title_size = 25
        label_size = 22
        offset_size = 15

        ## plot and save
        pic_folder = os.path.join('picture', 'Phase')
        os.makedirs(pic_folder, exist_ok=True)

        # input_path = "./Data/DMR25C2H4/DMR25C2H4_26wJetFlame_X.npy"
        # input_path = "./Data/DMR25C2H4/DMR25C2H4_266wMF_X.npy"
        # input_path = "./Data/DRM19/DRM19_78wCH4MF_X.npy"
        # input_path = "./Data/DRM19/DRM19_1kAdFlame_X.npy"
        # label_path = input_path.replace("X.npy", "Y.npy")
        # input1 = np.load(input_path)
        # label1 = np.load(label_path)

        fig = plt.figure(figsize=(12.8, 9.6))
        for dim in range(2 + self.gas.n_species):
            order_subplot = dim % 9 + 1
            ax = fig.add_subplot(3, 3, order_subplot)

            # points1 = ax.scatter(
            #     input1[:, dim],
            #     (label1[:, dim] - input1[:, dim]) / delta_t,
            #     c=input1[:, 0],  #colored by Temperature
            #     cmap='plasma',
            #     s=0.2,
            #     alpha=1)
                
            points = ax.scatter(
                input[:, dim],
                (label[:, dim] - input[:, dim]) / delta_t,
                c=input[:, 0],  #colored by Temperature
                cmap='rainbow',
                s=0.2,
                alpha=1)
            



            # if dim == 0:
            #     ax.set_yscale('log')
            #     ticks = ['$10^{%d}$' % i for i in range(2,9)]
            #     nums = [10**j for j in range(2,9)]
            #     plt.yticks(nums, ticks)
            # else:
            ax.ticklabel_format(style='sci', scilimits=(0, 3), axis='y')
            ax.xaxis.set_tick_params(labelsize=label_size)  #tick size
            ax.yaxis.set_tick_params(labelsize=label_size)
            ax.yaxis.get_offset_text().set_size(offset_size)
            # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
            plt.title(self._latexStyleName(dim),
                      fontsize=title_size,
                      fontweight='bold')

            ## savefig condition
            if order_subplot == 9:
                self._subplotPhaseSave(fig, plt, points, data_name, dim,
                                       pic_folder, dpi)
                fig = plt.figure(figsize=(12.8, 9.6))
            elif dim == 1 + self.gas.n_species:
                self._subplotPhaseSave(fig, plt, points, data_name, dim + 1,
                                       pic_folder, dpi)

    @staticmethod
    def _subplotPhaseSave(fig, plt, points, data_name, dim, pic_folder, dpi):
        r"""The built-in function to save phase diagram."""
        cbar_ax = fig.add_axes([0.98, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(points, orientation="vertical", cax=cbar_ax)
        cbar.set_label("$\\bf{Temperature (K)}$",
                       fontsize=24,
                       fontweight='bold')
        cbar.ax.tick_params(labelsize=20)
        fig.tight_layout(pad=2.5)
        pic_name = f'phase_of_{data_name}_{ceil(dim / 9)}.png'
        pic_path = os.path.join(pic_folder, pic_name)
        plt.savefig(pic_path, bbox_inches='tight', dpi=dpi)
        plt.close()
        print(f'phase picture saved in {pic_path}')

    def singleDisplot(self,
                      input,
                      data_name,
                      scale,
                      plot_dims='all',
                      ignore_negative=True,
                      size_show=500000,
                      offset=10,
                      dpi=200):
        r"""Draw and save the distribution diagram of a chemical dataset (TPY). If plot_dims='all', 
        Each figure contains 9 subplots. Else, each figure contains one subplot.
            
        Parameters
        ----------
        input : str or numpy.ndarray
            Input dataset which should be non-negative.
        data_name : str 
            The name of dataset, will be used when saving figures.
        scale : str
            Data scale, could be 'log', 'lin' or 'bct'.
        plot_dims : str or list, optional
            List of dimensions expected to be plotted, e.g. [1,4,5,7] or Range(20) or ['T','O','CH4','N2']. 
            Default string 'all' means plotting all the dimensions and every 9 subplots in one figure.
        ignore_negative : bool, optinal
            Whether ignore negative values in the input dataset. Default True.
        size_show : int, optional
            Random pick samples in dataset to show. Default 500,000.
        offset : int,optional
            Approximate display range for species under log scale. For instance, :math:`[10^{x}, 10^{x+offset}]`. Default 10.
        dpi : int
            The dpi used to save figure.

        """
        input = self._preCheckBeforePlot(input, scale, ignore_negative)
        ## shuffle datatset and randomly pick data
        permutation = np.random.permutation(input.shape[0])
        input = input[permutation]
        input = input[:size_show, :]

        if scale == 'bct':
            lamda = 0.1
            input[:, 2:] = (input[:, 2:]**(lamda) - 1) / lamda
            axis_labels = ['K', 'atm'] + [
                'mass fraction (bct)' for sp in self.gas.species_names
            ]
        else:
            axis_labels = ['K', 'atm'] + [
                'mass fraction' for sp in self.gas.species_names
            ]

        ## setup default font size and alpha
        alpha = 0.6
        title_size = 14
        label_size = 14
        ticks_size = 14

        ## plot
        pic_fold = os.path.join('picture', 'Distribution')
        os.makedirs(pic_fold, exist_ok=True)
        index_N2 = self.gas.species_index('N2') + 2
        linear_dims = [0, 1, index_N2]  # linear scale dims

        # 9 subplot in one figure, or one subpot in one figure
        fig = plt.figure(
            figsize=(12.8, 9.6)) if plot_dims == 'all' else plt.figure()

        ## check dims expected to display
        names = ['T', 'P'] + self.gas.species_names  # ['T','P',species_names]
        n_dims = self.gas.n_species + 2
        if plot_dims == 'all':
            dims = range(n_dims)
        else:
            dims = plot_dims

        ##
        for dim in dims:
            if isinstance(dim, str):
                if dim in ['T', 'P']:
                    dim = names.index(dim)
                else:
                    dim = self.gas.species_index(dim) + 2
            if plot_dims == 'all':  #plot all
                order_subplot = dim % 9 + 1
                ax = plt.subplot(3, 3, order_subplot)
            else:
                order_subplot = dim % 1 + 1
                ax = plt.subplot(1, 1, order_subplot)
            data = input[:, dim]
            ## subplot setup
            title = self._latexStyleName(dim)
            plt.title(title, fontsize=title_size, fontweight='bold')
            plt.xlabel(axis_labels[dim], fontsize=label_size)
            plt.ylabel("probability density", fontsize=label_size)
            plt.yticks(fontsize=ticks_size)
            plt.grid(alpha=0.3)  # add grid

            ## displot
            if scale == 'log' and dim not in linear_dims:
                sns.distplot(np.log10(data + 1e-40),
                             bins=1000,
                             kde=True,
                             color='seagreen',
                             hist_kws={"alpha": alpha})
                ## todo: adjust display range under log scale
                degree_lower, degree_upper = self._logDegree(data, offset)
                nums = [i for i in range(degree_lower, degree_upper + 1)]
                ticks = [
                    '$10^{%d}$' % i
                    for i in range(degree_lower, degree_upper + 1)
                ]
                plt.xticks(nums, ticks, fontsize=ticks_size)
                plt.xlim(degree_lower, degree_upper)
                # ax.set_xscale('log')
                ticker_base = ceil((degree_upper - degree_lower) / 5)
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(base=ticker_base))
            else:
                sns.distplot(data,
                             bins=1000,
                             kde=True,
                             color='seagreen',
                             hist_kws={"alpha": alpha})
                plt.xticks(fontsize=ticks_size)
            ## set save condition
            if plot_dims == 'all':
                if order_subplot == 9:
                    self._subplotDisSave(plt, pic_fold, data_name, scale, dim,
                                         dpi)
                    fig = plt.figure(figsize=(12.8, 9.6))
                elif dim == 1 + self.gas.n_species:
                    self._subplotDisSave(plt, pic_fold, data_name, scale,
                                         dim + 1, dpi)
            else:
                self._subplotDisSave(plt, pic_fold, data_name + '_dim', scale,
                                     dim * 9, dpi)
                fig = plt.figure()

        print(f'displot takes data size {input.shape}')

    @staticmethod
    def _subplotDisSave(plt, pic_fold, data_name, scale, dim, dpi):
        r"""The built-in function to save distribution."""
        plt.tight_layout()
        pic_name = f'displot_of_{data_name}_{ceil(dim / 9)}_scale={scale}.png'
        pic_path = os.path.join(pic_fold, pic_name)
        plt.savefig(pic_path, dpi=dpi)
        plt.close()
        print(f'displot picture saved in {pic_path}')

    def doubleDisplot(self,
                      input1,
                      input2,
                      label_text1,
                      label_text2,
                      data_name,
                      scale,
                      plot_dims='all',
                      ignore_negative=True,
                      size_show=500000,
                      offset=10,
                      dpi=200):
        r"""Draw and save the distribution diagram of two chemical datasets (TPY). If plot_dims='all', 
        Each figure contains 9 subplots. Else, each figure contains one subplot.
            
        Parameters
        ---------
        input1 : str or numpy.ndarray
            Dataset 1 which should be non-negative.
        input2 : str or numpy.ndarray
            Dataset 2 which should be non-negative.
        label_text1 : str
            Legend of dataset 1.
        label_text2 : str 
            Legend of dataset 2.
        data_name : str
            The datasets name, will be used when saving figures.
        scale : str
            Data scale, could be 'log', 'lin' or 'bct'.
        plot_dims : str or list, optional 
            List of dimension expected to be plotted, e.g. [1,4,5,7] or Range(20) or ['T','O','CH4','N2'].
            Default string 'all' means plotting all the dimensions and every 9 subplots in one figure.
        ignore_negative : bool, optional
            Whether ignore negative values in dataset. Default True.
        size_show : int, optional
            Random pick samples in dataset to show. Default 500,000.
        offset : int,optional
            Approximate display range for species under log scale. For instance, :math:`[10^{x}, 10^{x+offset}]`. Default 10.
        dpi : int
            The dpi used to save figure.


        """
        ## check input class type
        input1 = self._preCheckBeforePlot(input1, scale, ignore_negative)
        input2 = self._preCheckBeforePlot(input2, scale, ignore_negative)

        ## shuffle datatset and randomly pick data
        n_row = input1.shape[0]
        pick = np.random.choice(n_row,
                                size=min(size_show, n_row),
                                replace=False)
        input1 = input1[pick, :]

        n_row = input2.shape[0]
        pick = np.random.choice(n_row,
                                size=min(size_show, n_row),
                                replace=False)
        input2 = input2[pick, :]

        if scale == 'bct':
            lamda = 0.1
            input1[:, 2:] = (input1[:, 2:]**(lamda) - 1) / lamda
            input2[:, 2:] = (input2[:, 2:]**(lamda) - 1) / lamda
            axis_labels = ['K', 'atm'] + [
                'mass fraction (bct)' for sp in self.gas.species_names
            ]
        else:
            axis_labels = ['K', 'atm'] + [
                'mass fraction' for sp in self.gas.species_names
            ]

        ## setup default font size and alpha
        alpha1, alpha2 = 0.5, 0.3
        title_size = 14
        label_size = 14
        ticks_size = 14
        legend_size = 14

        ## plot
        pic_fold = os.path.join('picture', 'Distribution')
        os.makedirs(pic_fold, exist_ok=True)
        plt.figure(figsize=(12.8, 9.6))
        index_N2 = self.gas.species_index('N2') + 2
        linear_dims = [0, 1, index_N2]  #linear scale dims

        # 9 subplot in one figure, or one subpot in one figure
        fig = plt.figure(
            figsize=(12.8, 9.6)) if plot_dims == 'all' else plt.figure()

        ## check dims expected to display
        names = ['T', 'P'] + self.gas.species_names  # ['T','P',species_names]
        n_dims = self.gas.n_species + 2
        if plot_dims == 'all':
            dims = range(n_dims)
        else:
            dims = plot_dims

        for dim in dims:
            if isinstance(dim, str):
                if dim in ['T', 'P']:
                    dim = names.index(dim)
                else:
                    dim = self.gas.species_index(dim) + 2
            if plot_dims == 'all':  #plot all
                order_subplot = dim % 9 + 1
                ax = plt.subplot(3, 3, order_subplot)
            else:
                order_subplot = dim % 1 + 1
                ax = plt.subplot(1, 1, order_subplot)

            data1 = input1[:, dim]
            data2 = input2[:, dim]
            ## subplot setup
            title = self._latexStyleName(dim)
            plt.title(title, fontsize=title_size, fontweight='bold')
            plt.xlabel(axis_labels[dim], fontsize=label_size)
            plt.ylabel("probability density", fontsize=label_size)
            plt.yticks(fontsize=ticks_size)
            # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f')) # xlabel format
            plt.grid(alpha=0.3)  # add grid

            ## displot
            if scale == 'log' and dim not in linear_dims:
                sns.distplot(np.log10(data1 + 1e-40),
                             bins=1000,
                             kde=True,
                             label=label_text1,
                             color='seagreen',
                             hist_kws={"alpha": alpha1})
                sns.distplot(np.log10(data2 + 1e-40),
                             bins=1000,
                             kde=True,
                             label=label_text2,
                             color='chocolate',
                             hist_kws={"alpha": alpha2})
                plt.legend(loc='upper right', fontsize=legend_size)
                ## todo: adjust display range under log scale
                degree_lower, degree_upper = self._logDegree(data1, offset)
                nums = [i for i in range(degree_lower, degree_upper + 1)]
                ticks = [
                    '$10^{%d}$' % i
                    for i in range(degree_lower, degree_upper + 1)
                ]
                plt.xticks(nums, ticks, fontsize=ticks_size)
                plt.xlim(degree_lower, degree_upper)
                # ax.set_xscale('log')
                ticker_base = ceil((degree_upper - degree_lower) / 5)
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(base=ticker_base))
            else:
                # if scale=='lin':
                # ax.ticklabel_format(style='sci', scilimits=(0,1), axis='x')
                sns.distplot(data1,
                             bins=1000,
                             kde=True,
                             label=label_text1,
                             color='seagreen',
                             hist_kws={"alpha": alpha1})
                sns.distplot(data2,
                             bins=1000,
                             kde=True,
                             label=label_text2,
                             color='chocolate',
                             hist_kws={"alpha": alpha2})
                plt.xticks(fontsize=ticks_size)
                plt.legend(loc='upper right', fontsize=legend_size)
            ## set save condition
            if plot_dims == 'all':
                if order_subplot == 9:
                    self._subplotDoubleDisSave(plt, pic_fold, data_name, scale,
                                               dim, dpi)
                    fig = plt.figure(figsize=(12.8, 9.6))
                elif dim == 1 + self.gas.n_species:
                    self._subplotDoubleDisSave(plt, pic_fold, data_name, scale,
                                               dim + 1, dpi)
            else:
                self._subplotDoubleDisSave(plt, pic_fold, data_name + '_dim',
                                           scale, dim * 9, dpi)
                fig = plt.figure()
        print(
            f'double-displot takes data size {input1.shape} and {input2.shape}'
        )

    @staticmethod
    def _subplotDoubleDisSave(plt, pic_fold, data_name, scale, dim, dpi):
        r"""The built-in function to save double dataset distribution diagram."""
        plt.tight_layout()
        pic_name = f'double-displot_of_{data_name}_{ceil(dim / 9)}_scale={scale}.png'
        pic_path = os.path.join(pic_fold, pic_name)
        plt.savefig(pic_path, dpi=dpi)
        plt.close()
        print(f'double-displot picture saved in {pic_path}')

    @staticmethod
    def _preCheckBeforePlot(input, scale, ignore_negative):
        r"""Check dataset whether contain negative values. If there exists negative values in the dataset,
        raise warning. If scale is not in ['log','lin','bct'], raise TypeError.

        Parameters
        ---------
            input : str or numpy.ndarray
                The chemical dataset. If input is str, then input=np.load(input).
            scale : str
                Data scale, could be 'lin', 'log' or 'bct'.
            ignore_negative : bool
                Whether ignore negative values in dataset.

        """
        if isinstance(input, np.ndarray):
            pass
        elif isinstance(input, str):
            input = np.load(input)
        else:
            raise TypeError(
                f"expected input be <class 'str'> or <class 'numpy.ndarray'> but got {input.__class__}"
            )

        ## check scale
        if not scale in ['bct', 'lin', 'log']:
            raise TypeError(
                f"expected scale be either 'log','lin' or 'bct' but got {scale}"
            )

        ## pre-process if scale='bct'
        # if not ignore_negative:
        negative = np.where(input < 0)[0]
        negative = np.unique(negative)
        if negative.shape[0]:
            msg = 'expected non-negative value but got negative'
            # raise ValueError()
            warnings.warn(msg)
        if ignore_negative:
            input = np.delete(input, negative, axis=0)
        return input

    @staticmethod
    def _logDegree(data, offset):
        r"""Determine the approximate range of species mass fraction. The range will be used to display mass fraction under log scale. 

        Parameters
        ----------
        data : numpy.ndarray
            Species mass fraction which should be in [0,1].
        
        offset : int 
            Desired display range for species under log scale. 
        
        Returns
        -------
            low : int
                Let `mid` be the median of :math:`log_{10}(data)` and then **low** will be `mid-offset/2`.
            up : int
                Let `mid` be the median of :math:`log_{10}(data)` and then **up** will be `mid+offset/2`.
        
        """
        mid = np.median(np.log10(np.abs(data) + 1e-40))
        minimum = floor(np.min(np.log10(np.abs(data) + 1e-40)))
        low = max(minimum - 2, floor(mid - offset / 2))  #int
        up = min(0, ceil(mid + offset / 2))
        return low, up

    @staticmethod
    def _latexStyle(text):
        r"""Convert the given text to :math:`\LaTeX` bold style e.g. T-->$\\bf{T}$, CH4-->$\\bf{CH_{4}}$, C10H18O8-->$\\bf{C_{10}H_{18}O_{8}}$ """
        content = re.findall('\w\d+',
                             text)  # find letter+number e.g. C10 H2 N8 O12...
        content = list(set(content))  #unique
        for letter_number in content:
            letter = letter_number[0]
            number = letter_number[1:]
            text = re.sub(letter_number, f'{letter}_' + '{' + number + '}',
                          text)
        return '$\\bf{' + text + '}$'

    def _latexStyleName(self, index):
        r"""For a chemical dataset (TPY), the state names could be denoted as ['T','P',...species names]. Convert state_names[index] to 
        :math:`\LaTeX` bold style if there exists :math:`\LaTeX` enviroment.
        """
        state_names = ['T', 'P'] + self.gas.species_names
        # name = self._latexStyle(state_names[index]) if find_executable(
            # 'latex') else state_names[index]
        name = state_names[index]
        return name