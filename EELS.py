import os, sys
from random import randint
import copy

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.optimize import leastsq

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.colors import ListedColormap

#import seaborn as sns
#from random import shuffle

class Spectrum:
    def __init__(self):
        self.info = {} # Information dictionary
        #self.set_dir()


    def add_data(self):
        self.info['filename']   = False
        self.info['input']      = False
        self.info['window_len'] = False
        self.info['beta']       = False
        #self.info['data']       = False
        #self.info['path']       = False
        
    def set_dir(self, directory=None):
        if directory is None:
            path = os.path.dirname(os.path.realpath(__file__))
            self.info['path'] = path
            os.chdir(path)
        else:
            path = directory
            self.info['path'] = path
            os.chdir(path)

    def read_input(self, name, filename):
        self.add_data()
        #self.set_dir()
        self.name = name
        self.info['filename'] = filename
        self.info[name] = False

        try:
            path = self.info['path']
        except:
            print("""You need to specify current file directory before calling read_input.""")
            print("""Usage: (Spectrum type).set_dir(directory_name)""")
            print("""Error: Reading input file aborted.\n""")
            sys.exit()

        if not os.path.isdir('input'):
            print("""Warning: There is no input file, 'input.txt'.""")
            print("""The filtering function will adopt default settings.""")
            """sys.exit()"""
        else:
            os.chdir(os.path.join(path, 'input'))
            self.input_path = os.path.join('input', 'input.txt')
            with open(self.input_path, 'r') as _file:
                _input = _file.readlines()
            self.info['input'] = True


        window_len = beta = 0
        if self.info['input']:
            for i in range(0, len(_input)):
                if _input[i][0] == '$':
                    continue
                elif "$" in _input[i]:
                    for j in range(0, len(_input[i])):
                        if "$" == _input[i][j]:
                            end = j

                    if "channel" in _input[i][:end]:
                        for j in range(0, len(_input[i])):
                            if "=" == _input[i][j]:
                                file_num = eval(_input[i][j+1:end])
                                window_len = file_num
                    elif "beta" in _input[i][:end]:
                        for j in range(0, len(_input[i])):
                            if "=" == _input[i][j]:
                                file_num = eval(_input[i][j+1:end])
                                beta = file_num
                else:
                    if "channel" in _input[i]:
                        for j in range(0, len(_input[i])):
                            if "=" == _input[i][j]:
                                file_num = eval(_input[i][j+1:])
                                window_len = file_num
                    elif "beta" in _input[i]:
                        for j in range(0, len(_input[i])):
                            if "=" == _input[i][j]:
                                file_num = eval(_input[i][j+1:])
                                beta = file_num
                                
            if not (int(window_len) % 2) == 0:
                print("Channel length for low-pass filter should be even number.")
                sys.exit()
            self.info['window_len'] = int(window_len)
            self.info['beta']       = int(beta)
        else:
            self.info['window_len'] = 10
            self.info['beta']       = 5 #Default setting for filterfing

        if os.path.isdir('input'):
            _file.close()
            #os.chdir(os.path.join(path, 'input'))


        data = np.zeros([1])
        try:
            data = np.loadtxt(filename)
            """The input spectrum file should have
            column1: energy loss
            column2: intensity"""
            self.info[name] = copy.deepcopy(data)
        except:
            print('The file, '+filename+', cannot be read by NUMPY due to its data structure.')

        if not type(self.info[name]).__module__ == 'numpy':
            print("""There is no readable input files in the input folder.""")
            sys.exit()


    def output(self, name, filename, option=False, col_name=None):
        """
        If you want to specify output file directory, you should use set_dir function.

        If option = false, this function will not write column names.
        If option = true , this function will     write column names.
        """
        
        path = self.info['path']
        os.chdir(path)
        if os.path.isdir('output') == False:
            os.mkdir('output')
            os.chdir(os.path.join(path,'output'))
        else:
            os.chdir(os.path.join(path,'output'))


        #out = open(os.path.splitext(self.file_list[i])[0]+'_output.txt','w')
        out = open(filename,'w')
        
        if(option):
            statement = ''
            statement += '{:<30}'.format('Energy Loss')
            if(col_name==None):
                print("Warning: no col_name in output method. You should designate col_name if option = True")
            statement += '\t{:<30}'.format(col_name)
            out.write(statement+'\n')
        
        data = copy.deepcopy(self.info[name][:,:])
        for i in range(0, data.shape[0]):
            statement = '{:<30}\t{:<30}\n'.format(str(data[i,0]), str(data[i,1]))
            out.write(statement)
        out.close()
        
        #os.chdir(os.path.join(path, 'input'))
        os.chdir(path)


    def all_output(self, args):
        """
        If you want to specify output file directory, you should use set_dir function.

        Filename should be defined at the last arg.
        """
        path = self.info['path']
        os.chdir(path)
        if os.path.isdir('output') == False:
            os.mkdir('output')
            os.chdir(os.path.join(path,'output'))
        else:
            os.chdir(os.path.join(path,'output'))

        out = open(args[-1],'w')


        for j in range(0, len(args[0:-1])):
            if not isinstance(args[j], str):
                print("""Arguments should be defined as strings.""")
                sys.exit()

        data = []
        for j in range(0, len(args[0:-1])):
            data.append(self.info[args[j]][:,:])

#        dim = data[0].shape[0]
#        for j in range(1, len(data)):
#            if not dim == data[j].shape[0]:
#                print("""All output files should have same number of channels.""")
#                sys.exit()
        num = 0
        for j in range(0, len(data)):
            temp = data[j].shape[0]
            if temp >= num:
                num = temp
        
        temp = []
        for j in range(0, len(data)):
            dim = data[j].shape[0]
            temp.append(np.zeros([num,2]))
            if not dim == num:
                temp[j][:dim,:] = copy.deepcopy(data[j][:,:])
                temp[j][dim:,:] = -300000
            else:
                temp[j][:,:] = copy.deepcopy(data[j][:,:])
                
        data = []
        for j in range(0, len(temp)):
            data.append(temp[j][:,:])
        dim = data[0].shape[0]

        statement = ''
        for i in range(0, len(data)):
            statement += '{:<30}\t{:<30}\t'.format('Energy loss', args[i])
        out.write(statement+'\n')

        for j in range(0, dim):
            statement = ''
            for i in range(0, len(data)):
                if data[i][j,0] < -200000:
                    statement += '{:<30}\t{:<30}\t'.format('', '')
                else:
                    statement += '{:<30}\t{:<30}\t'.format(str(data[i][j,0]), str(data[i][j,1]))
            out.write(statement+'\n')
        out.close()
        
        #os.chdir(os.path.join(path, 'input'))
        os.chdir(path)


    def low_pass_filter(self, name, filtername):
        """
        @author: Subin Bang
        @institution: Seoul National University, 2016

        You need numpy, scipy, and matplotlib packages to be installed with python version 3.x or higher.
        Store an input file(input.txt) and data sets in input folder.
        input.txt: channel=... \n beta = ...
        This script will use Kaiser-window function to add FIR filter.
        Kaiser window will need two input parameters, number of channels and beta.

        Number of channels indicates the width of filtered peaks.
        This number should be EVEN.
        1 channel = resolution of the data
        Example) If your data has 2000 energy loss points and the spectrum was
                 recorded from 0 eV to 100 eV, 1 channel = (100-0)/2000 = 0.05 eV.
                 If number_of_channels=10, peaks narrower than 0.5 eV would be neglected.
                 Too large number will remove the very detail of the original spectrum.
                 Too small number will have noisy problems. Try 30 channels first.

                 beta is related with the shape of the window funtion.
                 0	 Rectangular
                 5	 Similar to a Hamming
                 6	 Similar to a Hanning
                 8.6	 Similar to a Blackman
                 Standard numpy library recommends for you to use 14 first.
        """
        _filtered_data = []
        data = copy.deepcopy(self.info[name][:,:])
        window_len = self.info['window_len']
        beta = self.info['beta']
        w = np.kaiser(window_len, beta)


        #_filtered_data.append(np.r_[self._data[i][window_len-1:0:-1,1],self._data[i][:,1],self._data[i][-1:-window_len:-1,1]])
        _filtered_data = np.r_[data[window_len-1:0:-1,1],data[:,1],data[-1:-window_len:-1,1]]
        _convolved = np.convolve(w/w.sum(),_filtered_data,mode='valid')
        _signal = _convolved[int(window_len/2)-1:-(int(window_len/2))]
        row = (data.shape)[0]
        col = (data.shape)[1]
        self.info[filtername] = np.zeros([row,col])
        self.info[filtername][:,0] = data[:,0]
        self.info[filtername][:,1] = copy.deepcopy(_signal)


    def overlap_filter(self, name, filtername, start, end, channel):
        """Data should be numpy.ndarray form of
        column1: energy loss
        column2: intensity

        start & end: start/endpoint (energy loss) of data to make smooth
        channel: minimum distance between local maxima/minima, > filtering channel
        Example: test.overlap(test._filtered_data[0],2.5,18.2104,15)
        If data is not filtered, the result will be very poor.
        If end = 'end', the function will automatically search the endpoint."""

        data = copy.deepcopy(self.info[name][:,:])
        dispersion = self.info[self.name][1,0] - self.info[self.name][0,0]
        loc_start = np.min(np.where(np.absolute(data[:,0]-start)<dispersion))
        if end == 'end':
            loc_end = data.shape[0]
        else:
            loc_end   = np.min(np.where(np.absolute(data[:,0]-end)<dispersion))

        temp = copy.deepcopy(data[loc_start:loc_end+1,:])

        #peakind = signal.find_peaks_cwt(temp, np.arange(3,20)) # should read ref. paper in scipy.signal.find_peaks_cwt
        #stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        uppeakind = np.r_[True, temp[1:,1] > temp[:-1,1]] & np.r_[temp[:-1,1] > temp[1:,1], True]
        uppeakind[0] = uppeakind[-1] = True
        downpeakind = np.r_[True, temp[1:,1] < temp[:-1,1]] & np.r_[temp[:-1,1] < temp[1:,1], True]
        downpeakind[0] = downpeakind[-1] = True

        for i in range(0,(uppeakind.shape)[0]):
            if not uppeakind[i]:
                continue
            for j in range(i+1,(uppeakind.shape)[0]):
                if ((j-i) < channel) and uppeakind[j]:
                    uppeakind[j] = False

        for i in range(0,(downpeakind.shape)[0]):
            if not downpeakind[i]:
                continue
            for j in range(i+1,(downpeakind.shape)[0]):
                if ((j-i) < channel) and downpeakind[j]:
                    downpeakind[j] = False

        uppeakind[0] = uppeakind[-1] = True
        downpeakind[0] = downpeakind[-1] = True
        f_up   = interp1d(temp[uppeakind,0],temp[uppeakind,1], kind='cubic')
        f_down = interp1d(temp[downpeakind,0],temp[downpeakind,1], kind='cubic')

        overlapped = (f_up(temp[:,0]) + f_down(temp[:,0]))/2.
        """
        out = open('test.txt','w')
        for i in range(0,(overlapped.shape)[0]):
            out.write(str(overlapped[i])+'\n')
        out.close()
        """
        self.info[filtername] = np.zeros(data.shape)
        self.info[filtername][:,0] = copy.deepcopy(data[:,0])
        self.info[filtername][loc_start:loc_end+1,1] = copy.deepcopy(overlapped)
        self.info[filtername][0:loc_start,1] = copy.deepcopy(data[0:loc_start,1])
        if not end == 'end':
            self.info[filtername][loc_end+1:,1] = copy.deepcopy(data[loc_end+1:,1])


    def spline_filter(self, name, filtername, start, end, channel):
        
        dispersion = self.info[self.name][1,0] - self.info[self.name][0,0]
        loc_start = np.min(np.where(np.absolute(self.info[self.name][:,0]-start)<dispersion))
        loc_end   = np.min(np.where(np.absolute(self.info[self.name][:,0]-end)<dispersion))

        temp = copy.deepcopy(self.info[name][:,:])
        peakind = temp[:,0]
        peakind = (peakind==-2000)
        for i in np.arange(int(loc_start),int(loc_end),channel):
            peakind[i] = True
        peakind[int(loc_end)] = True

        """
        peakind[np.argmax(temp[int(loc_start):int(loc_end),1])+int(loc_start)] = False # Largest peak removed to make peak more smooth
        peakind[np.argmax(temp[int(loc_start):int(loc_end),1])+int(loc_start)-1] = False
        peakind[np.argmax(temp[int(loc_start):int(loc_end),1])+int(loc_start)+1] = False
        """
        try:
            f = interp1d(temp[peakind,0],temp[peakind,1], kind='cubic')
        except:
            print("Channel width is too small to make interpolation.")
            sys.exit()
        fpeak = f(temp[int(loc_start):int(loc_end)+1,0])
        temp[int(loc_start):int(loc_end)+1,1] = fpeak[:]

        self.info[filtername] = copy.deepcopy(temp)


    def smoothing_spline(self, name, filtername, start, end, p):
        """
        Smoothing spline method with parameter p,
        using de Boor's algorithm written in FORTRAN90.
        This code is Python version of de Boor's FORTRAN library
        and (nearly) equivalent to csaps function in MATLAB.

        Reference: pages.cs.wisc.edu/~deboor/
                   www.eng.mu.edu/frigof/spline.html
                   de Boor, A Practical Guide to Splines, 2001.

        p = 0 : least-square straight-line fit to data
        p = 1 : cubic spline interpolant
        Subin Bang, Seoul National University @ 2017.

        If
        start = 'start' : automatically search first-point data
        end   = 'end'   : automatically search end-point   data
        """
        data = copy.deepcopy(self.info[name][:,:])
        dispersion = data[1,0] - data[0,0]
        if start == 'start':
            loc_start = 0
        else:
            loc_start = int(np.min(np.where(np.absolute(data[:,0]-start)<dispersion)))

        if end == 'end':
            loc_end = -1
        else:
            loc_end = int(np.min(np.where(np.absolute(data[:,0]-end)<dispersion)))

        y = copy.deepcopy(data[loc_start:loc_end,1])
        npoint = y.shape[0]
        x = np.linspace(0.0, (npoint-1.)/npoint, num=npoint)


        """setup q"""
        v = np.zeros([npoint,7])
        v[0,3] = x[1]-x[0]
        for i in range(1,npoint-1):
            v[i,3] = x[i+1]-x[i]

            # Here, all var(y) = dy = delta_y in de Boor's algorithm would be equally set to 1.
            v[i,0] = 1./v[i-1,3]
            v[i,1] = ((-1.*1.)/v[i,3]) - (1./v[i-1,3])
            v[i,2] = 1./v[i,3]
        
        v[npoint-1,0] = 0.
        for i in range(1,npoint-1):
            v[i,4] = (v[i,0]*v[i,0]) + (v[i,1]*v[i,1]) + (v[i,2]*v[i,2])
        for i in range(2,npoint-1):
            v[i-1,5] = (v[i-1,1]*v[i,0]) + (v[i-1,2]*v[i,1])

        v[npoint-2,5] = 0.

        for i in range(3,npoint-1):
            v[i-2,6] = (v[i-2,2]*v[i,0])

        v[npoint-3,6] = 0.
        v[npoint-2,6] = 0.


        """Construct q-transp*y in qty"""
        prev = (y[1]-y[0])/v[0,3]
        a = np.zeros([npoint,4])
        for i in range(1,npoint-1):
            diff = (y[i+1]-y[i])/v[i,3]
            a[i,3] = diff - prev
            prev = diff


        """Construct 6*(1-p)*q-transp*(d**2)*q + p*r"""
        six1mp = 6.*(1.-p)
        twop   = 2.*p

        for i in range(1,npoint-1):
            v[i,0] = (six1mp*v[i,4]) + (twop*v[i-1,3] + v[i,3])
            v[i,1] = (six1mp*v[i,5]) + (p*v[i,3])
            v[i,2] = six1mp*v[i,6]


        """Factorization"""
        for i in range(1,npoint-2):
            ratio = v[i,1]/v[i,0]
            v[i+1,0] = v[i+1,0]-ratio*v[i,1]
            v[i+1,1] = v[i+1,1]-ratio*v[i,2]
            v[i,1] = ratio
            ratio = v[i,2]/v[i,0]
            v[i+2,0] = v[i+2,0]-ratio*v[i,2]
            v[i,2] = ratio
        

        """Forward Substitution"""
        a[0,2] = 0.
        v[0,2] = 0.
        a[1,2] = a[1,3]
        for i in range(1,npoint-2):
            a[i+1,2] = a[i+1,3] - (v[i,1]*a[i,2]) - (v[i-1,2]*a[i-1,2])


        """Backward Substitution"""
        a[npoint-1,2] = 0.
        a[npoint-2,2] = a[npoint-2,2] / v[npoint-2,0]
        rr = np.linspace(1,int(npoint-3),int(npoint-3))
        for j in rr[::-1]:
            i = int(j)
            a[i,2] = (a[i,2]/v[i,0]) - (a[i+1,2]*v[i,1]) - (a[i+2,2]*v[i,2])


        """Construct Q*U"""
        prev = 0.
        for i in range(1,npoint):
            a[i,0] = (a[i,2]-a[i-1,2])/v[i-1,3]
            a[i-1,0] = a[i,0]-prev
            prev = a[i,0]

        a[npoint-1,0] = -1.*a[npoint-1,0]


        filtered = np.zeros([npoint,1])
        for i in range(0,npoint):
            filtered[i,0] = y[i]-(6.*(1.-p)*a[i,0])

        data[loc_start:loc_end,1] = copy.deepcopy(filtered[:,0])
        self.info[filtername] = copy.deepcopy(data)
        
        
    def savitzky_golay(self, name, filtername, start, end, window_size, order, deriv):
        """
        Savitzky-Golay type filter function
        for equally spaced dataset.
        
        References:
            1) Numerical Recipes in C: the Art of Scientific Computing
               (www.wire.tu-bs.de/OLDWEB/mameyer/cmr/savgol.pdf)
            2) Scipy built-in function, scipy.signal.savgol_filter
            3) scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
            4) en.wikipedia.org/wiki/Savitzky–Golay_filter
    
        order = order of polynomial
        deriv = order of derivative of polynomial. For example, if deriv = 1, 
                the output data will be first-derivative of Savitzky-Golay filtered
                data. For pure smoothing, use deriv = 0.
        """
        data = copy.deepcopy(self.info[name][:,:])
        
        dispersion = data[1,0] - data[0,0]
        if start == 'start':
            loc_start = 0
        else:
            loc_start = int(np.min(np.where(np.absolute(data[:,0]-start)<dispersion)))

        if end == 'end':
            loc_end = -1
        else:
            loc_end = int(np.min(np.where(np.absolute(data[:,0]-end)<dispersion)))
        
        filtered = copy.deepcopy(data[loc_start:loc_end,1])
        
        #from math import factorial
        
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
        
        if (window_size%2 != 1):
            print("Savitzky Golay: window-size automatically adjusted by the nearest odd number.")
            window_size += window_size
            
        if (window_size < 1):
            print("Savitzky Golay: window-size automatically adjusted by the default value.")
            window_size = 3
        
        if (window_size < order + 2):
            print("Savitzky Golay: window-size automatically adjusted by considering order")
            if (order%2 == 1):
                window_size = order + 2
            else:
                window_size = order + 3
        
        filtered[:] = signal.savgol_filter(filtered[:],window_size,order,deriv)
    
            
        data[loc_start:loc_end,1] = copy.deepcopy(filtered[:])
        self.info[filtername] = copy.deepcopy(data)
        
    def normalize(self, name, normed_name, option=4, position=None):
        """
        option = 1: [0,1] normalization
        option = 2: Maximum value normalization
        option = 3: Normalization by intensity of certain value, specified by position
        option = 4: Normalization by integrated intensity
        """
        
        data = copy.deepcopy(self.info[name][:,:])
        dispersion = data[1,0] - data[0,0]
        
        if (option==1):
            minval = np.amin(data[:,1])
            data[:,1] -= minval
            maxval = np.amax(data[:,1])
            data[:,1] /= maxval
                
        elif (option==2):
            maxval = np.amax(data[:,1])
            data[:,1] /= maxval
                
        elif (option==3):
            if position is None:
                print("""Normalize: assign your position.""")
                sys.exit()
                
            loc_val = int(np.min(np.where(np.absolute(data[:,0]-position)<dispersion)))
            posval = data[loc_val,1]
            data[:,1] /= posval
            
        elif (option==4):
            temp_sum = np.sum(data[:,1])
            data[:,1] /= temp_sum
            
        else:
            print("""Normalize: check your option.""")
            sys.exit()
            
        self.info[normed_name] = copy.deepcopy(data[:,:])
        
    def color_map_value(self, choice=1, cycle_option=False):
        """
        <choice>
        1: blue   (6)
        2: red    (20)
        3: green  (12)
        4: purple (2)
        5: orange (19)
        6: yellow (17)
        7: brown  (23)
        8: gray   (24)
        9: black  (0)
        """
        if not ( isinstance(choice, int) ):
            choice = randint(1,9)
        
        if ( (choice < 1) ):
            choice = randint(1,9)
        if ( (choice > 9) ):
            choice = choice % 9
            choice += 1
            
        color_dict = {1:6,2:20,3:12,4:2,5:19,6:17,7:23,8:24,9:0}
    
        numb = 25
        #ColorBrewer Scale
        colortheme = plt.get_cmap('nipy_spectral')
    
        values = [i for i in range(numb)]
        cNorm = colors.Normalize(vmin=0,vmax=values[-1])
    
        scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=colortheme)
        colormap  = scalarMap.to_rgba(values[color_dict[choice]])
    
        return colormap
        
    def plot(self, args, scatteroption, filename=None, xlim=[None,None], ylim=[None,None], legendout=True, coloroption=True):
        """
        args = list containing strings of dictionary data.
        scatteroption = same dimension as args, but containing True/False.
                        if scatteroption(i) = True, plot of args(i) = scatter-type.
        x/ylim = x/yrange of plot; list type.
        legendout = Place legend out of graph
        coloroption = Use color_brewer colormap
        """
        
        """
        ncurves = len(args)
        values = range(ncurves)
        colortheme = plt.get_cmap('nipy_spectral')
        #colortheme = ListedColormap(sns.color_palette("hls", 8))
        cNorm = colors.Normalize(vmin=0,vmax=values[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=colortheme)
        """
        
        """
        ncurves = len(args)
        colortheme = plt.get_cmap('nipy_spectral')
        if (ncurves > 11):
            values = [i for i in range(ncurves)]
            cNorm = colors.Normalize(vmin=0,vmax=values[-1])
        else:
            values = [i for i in range(11)]
            cNorm = colors.Normalize(vmin=0,vmax=11)

        shuffle(values)
        scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=colortheme)
        """
        
        plt.rcParams["font.family"] = "sans-serif"
        fig = plt.figure(num=1,figsize=(3,3),dpi=400,facecolor='w',edgecolor='k')
        #fig.suptitle("Large Title",fontsize=15,fontweight='bold')
        #fig.subplots_adjust(top=0.9, bottom=0.08, left=0.15, right=0.9, hspace=0.3, wspace=0.1)
        
        ax1 = plt.subplot(111) #numrow,numcol,figure_number
        ax1.set_ylabel('Intensity (arb. unit)',fontsize=13)
        ax1.set_xlabel('Energy Loss (eV)',fontsize=13)
        
        #plt.xticks(np.arange(min(data.info[name][:,0]),max(data.info[name][:,0])+1,5))
        #ax1.set_xlim([min(data.info[name][:,0]),max(data.info[name][:,0])])
        #ax1.set_ylim([min(data.info[name][:,1]),max(data.info[name][:,1])])
        
        if ( isinstance(xlim[0], (int, float)) and isinstance(xlim[1], (int, float)) ):
            ax1.set_xlim(xlim)
            #Xt = np.arange(xlim[0], xlim[1], 4)
            Xt = [526,531,536,541,546,551]
            ax1.xaxis.set_ticks(Xt[0:-1])
        if ( isinstance(ylim[0], (int, float)) and isinstance(ylim[1], (int, float)) ):
            ax1.set_ylim(ylim)
            #Yt = np.arange(ylim[0], ylim[1], 0.0001)
            Yt = [0]
            ax1.yaxis.set_ticks(Yt[0:-1])
        
        ax1.get_yaxis().set_tick_params(direction='in',width=1,size=5,color='black',labelsize=8,labelcolor='black',right='on')#,labelright='on')
        ax1.get_xaxis().set_tick_params(direction='in',width=1,size=5,color='black',labelsize=8,labelcolor='black',top='on')
        
        count = 0
        color_count = 1
        for names in args:
            colorVal = self.color_map_value(color_count) #scalarMap.to_rgba(values[color_count])
            
            if ( coloroption ):
                if ( scatteroption[count] ):
                    ax1.plot(self.info[names][:,0],self.info[names][:,1],marker='o',ms=0.5,linestyle = 'None',mec='k',mfc='k')
                else:
                    ax1.plot(self.info[names][:,0],self.info[names][:,1],label=names,lw=0.5,c=colorVal)
                    color_count += 1
            else:
                if ( scatteroption[count] ):
                    ax1.plot(self.info[names][:,0],self.info[names][:,1],marker='o',ms=1,linestyle = 'None')
                else:
                    ax1.plot(self.info[names][:,0],self.info[names][:,1],label=names,lw=0.5)
                
            count = count + 1
        
        handles, labels = ax1.get_legend_handles_labels()
        
        if (legendout):
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax1.legend(handles, labels, loc='upper right')

        
        fig.tight_layout()

        if not (filename is None):
            path = self.info['path']

            os.chdir(path)
            if os.path.isdir('output_plot') == False:
                os.mkdir('output_plot')
                os.chdir(os.path.join(path,'output_plot'))
            else:
                os.chdir(os.path.join(path,'output_plot'))
            
            if (legendout):
                plt.savefig(filename,bbox_inches='tight')
            else: # redundant...
                plt.savefig(filename,bbox_inches='tight')
                
            os.chdir(path)
        
        plt.show()

    def reflected_tail(self, name, zlp, inelastic):
        """
        Data with 'name' should have zero-loss peak.
        The peak is assumed to be the highest one.
        """
        data = copy.deepcopy(self.info[name][:,:])
        i0 = data[:,1].argmax(0)

        """
        FWQM from the first spectrum in channels
        Search only 2eV around the max to avoid counting the plasmons
        in thick samples
        """
        dispersion = self.info[self.name][1,0] - self.info[self.name][0,0]
        i_range = int(round(2. / dispersion))
        fwqm_bool = data[i0-i_range:i0+i_range,1] > 0.25 * data[i0,1]
        ch_fwqm = len(fwqm_bool[fwqm_bool])
        canvas = np.zeros(data[:,1].shape)

        """Reflect the tail"""
        width = int(round(1.5 * ch_fwqm))
        canvas[i0 + width : 2 * i0 + 1] = data[i0 - width::-1,1]

        """Remove the "background" = mean of first 4 channels and reflects the tail"""
        bkg = np.mean(data[0:4,1])
        canvas -= bkg

        """
        Scale the extended tail with the ratio obtained from
        2 overlapping channels
        """
        ch = i0 + width
        #ratio = np.mean(self.zero_loss[ch: ch + 2] / canvas[ch: ch + 2], 0)
        ratio = np.mean(data[ch:ch+3,1])/np.mean(canvas[ch:ch+3])
        canvas[:] *= ratio

        """Copy the extension"""
        canvas[canvas<0.] = 0.
        data[i0 + width:,1] = canvas[i0 + width:]

        temp = copy.deepcopy(self.info[name])
        self.info[zlp] = np.zeros(temp.shape)
        self.info[zlp][:,0] = copy.deepcopy(temp[:,0])
        self.info[zlp][:,1] = copy.deepcopy(data[:,1])

        self.info[inelastic] = np.zeros(temp.shape)
        self.info[inelastic][:,0] = copy.deepcopy(temp[:,0])
        self.info[inelastic][:,1] = copy.deepcopy(temp[:,1]-data[:,1])
        self.info[inelastic][self.info[inelastic][:,1]<0.,1] = 0.


    def gauss_lorentz(self, name, zlp, inelastic):
        """
        Data with 'name' should have zero-loss peak.
        The peak is assumed to be the highest one.
        Extract zero-loss peak by one Gaussian + one Lorentzian
        """
        data = copy.deepcopy(self.info[name][:,:])
        i0 = data[:,1].argmax(0)
        dispersion = self.info[self.name][1,0] - self.info[self.name][0,0]
        i_range = int(round(2. / dispersion))
        fwqm_bool = data[i0-i_range:i0+i_range,1] > 0.25 * data[i0,1]
        ch_fwqm = len(fwqm_bool[fwqm_bool])
        canvas = np.zeros(data[:,1].shape)

        """Reflect the tail"""
        width = int(round(1.5 * ch_fwqm))
        canvas[i0 + width : 2 * i0 + 1] = data[i0 - width::-1,1]
        ch = i0 + width
        ratio = np.mean(data[ch:ch+3,1])/np.mean(canvas[ch:ch+3])
        canvas[:] *= ratio
        canvas[0:ch] = data[0:ch,1]

        #from numpy import linalg as LA
        def gauss(x,p): # p[0]=mean, p[1]=area, p[2]=fwhm, p[3]=background
            return p[1] * np.exp(-(x- p[0])**2/(2.0*(p[2]/2.35482)**2)) #+ p[3]

        errfunc = lambda p, x, y: (gauss(x, p) - y)
        guess = [data[i0,0], data[i0,1]-data[0,1], 0.15, data[0,1]]
        p0 = np.array(guess)
        p1, success = leastsq(errfunc, p0[:], \
                              args=(data[0:2*i0+1,0],canvas[0:2*i0+1]), maxfev=200000)


        errfunc = lambda p, x, y: (self.conf(p, x) - y)
        guess = [p1[0], 0.8*p1[1], 0.2*p1[1], p1[2], p1[2]]
        p0 = np.array(guess)
        p1, success = leastsq(errfunc, p0[:], \
                              args=(data[0:2*i0+1,0],canvas[0:2*i0+1]), maxfev=200000)

        self.info[zlp] = np.zeros(data.shape)
        self.info[zlp][:,1] = copy.deepcopy(self.conf(p1[:], data[:,0]))
        self.info[zlp][:,0] = copy.deepcopy(data[:,0])
        self.info[inelastic] = np.zeros(data.shape)
        self.info[inelastic][:,1] = data[:,1] - self.info[zlp][:,1]
        self.info[inelastic][:,0] = data[:,0]

        """Consider -1 ~ 1eV among SI
        endx = int(startx+widthx)
        startx = int(startx-widthx)
        x = copy.deepcopy(data[startx:endx,0])
        y = copy.deepcopy(data[startx:endx,1])

        A = self.info[name][0,1]
        guess = [0.15, 1000, 0.01, \
                 0.15, 1000, 0.2 , \
                 0.15, 1000, 0.01, \
                 ]
        p0 = np.array(guess)

        guess[0] = FWHM     of Lorentzian1
        guess[1] = height   of Lorentzian1(area)
        guess[2] = position of Lorentzian1
        guess[3] = FWHM     of Lorentzian2 & Gaussian
        guess[4] = height   of Lorentzian2 & Gaussian(area)
        guess[5] = position of Lorentzian2 & Gaussian
        """



    def conf1(self,p,x,y):
        # p[0]=mean, p[1]=area Gauss, p[2]=area Lorentz
        # p[3]=fwhm Gauss, p[4]=fwhm Lorentz, p[5]=background
        #dispersion = self.info[self.name][1,0] - self.info[self.name][0,0]

        gauss    = p[1]*np.exp(-(x-p[0])**2/(2.*(p[3]/2.35482)**2))
        lorentz  = ( (0.5*p[4]*p[2]/3.141592) / ( (x-p[0])**2 + (p[4]/2)**2 ) )

        temp = gauss[:]+lorentz[:]-y[:]+7000
        #temp[temp>0] *= 100

        #return gauss+lorentz#+p[5]
        return temp

    def conf(self,p,x):
        # p[0]=mean, p[1]=area Gauss, p[2]=area Lorentz
        # p[3]=fwhm Gauss, p[4]=fwhm Lorentz, p[5]=background
        #dispersion = self.info[self.name][1,0] - self.info[self.name][0,0]

        gauss    = p[1]*np.exp(-(x-p[0])**2/(2.*(p[3]/2.35482)**2))
        lorentz  = ( (0.5*p[4]*p[2]/3.141592) / ( (x-p[0])**2 + (p[4]/2)**2 ) )

        return gauss+lorentz

    ###################### Adopted from Quantfit, spectrum.py ######################
    #Quantfit Start
    def extract_zero_loss(self, name, zlp, inelastic):
        data = copy.deepcopy(self.info[name][:,:])
        dispersion = data[1,0] - data[0,0]
        startx = data[:,1].argmax(0)

        widthx = int(abs(1./dispersion))

        endx = int(startx+widthx)
        startx = int(startx-widthx)
        x = copy.deepcopy(data[startx:endx,0])
        y = copy.deepcopy(data[startx:endx,1])

        # p[0] = Gaussian Height  , p[1] = Gaussian Position  , p[2] = Gaussian FWHM
        # p[3] = Lorentzian Height, p[4] = Lorentzian Position, p[5] = Lorentzian FWHM
        # p[6] = Lorentzian Height, p[7] = Lorentzian Position, p[8] = Lorentzian FWHM
        FWHM = 1.0
        guess = [data[startx+widthx,1], 0., 0.]
        p0 = np.array(guess)
        # Under Construction...


    def Zlfunc(self, name, p, x):
        dispersion = self.info[name][1,0] - self.info[name][0,0]

        # p[0] = Gaussian Height  , p[1] = Gaussian Position  , p[2] = Gaussian FWHM
        # p[3] = Lorentzian Height, p[4] = Lorentzian Position, p[5] = Lorentzian FWHM
        # p[6] = Lorentzian Height, p[7] = Lorentzian Position, p[8] = Lorentzian FWHM
        gauss =  p[0] * np.exp(-(x- p[1])**2 / (2.0*(p[2]/2.3548)**2))
        lorentz = ((0.5 *  p[3]* p[5]/3.14)/((x- p[4])**2+(( p[5]/2)**2)))
        lorentz2 = ((0.5 *  p[6]* p[7]/3.14)/((x- (p[7]))**2+(( p[7]/2)**2)))

        GP = np.array(gauss).argmax(0)
        gauss1 = np.roll(gauss, -GP+int((len(gauss)+0.5)/2.0))

        y = np.convolve((gauss1), lorentz*lorentz2, 'same')

        return y

    def doSSD(self, name):
        return
    #Quantfit End
    ################################################################################


    def fourier_log_deconvolution(self, spec, zlp, filtername):
        data = copy.deepcopy(self.info[spec][:,:])
        ZL   = copy.deepcopy(self.info[zlp][:,:])
        z = np.fft.rfft(ZL,axis=0)
        j = np.fft.rfft(data,axis=0)
        j1 = z * np.nan_to_num(np.log(j/z))

        sdata = np.fft.irfft(j1,axis=0)
        self.info[filtername] = copy.deepcopy(sdata)

#i = 0
#aa = []
#test = Spectrum()
#for files in os.listdir():
#    if files == 'input.txt':
#        continue
#    if 'ep2' in files:
#        test.read_input(files,files)
#        test.low_pass_filter(files,files+'_fil')
#        test.overlap_filter(files+'_fil',files+'_over',0.8, 'end', 2)
#        loc = np.min(np.where(np.absolute(test.info[files][:,0]-18)<0.01))
#        test.info[files][:,1] /= test.info[files][loc,1]
#        test.info[files+'_fil'][:,1] /= test.info[files+'_fil'][loc,1]
#        test.info[files+'_over'][:,1] /= test.info[files+'_over'][loc,1]
#
#        test.info[files][:,1] += i
#        test.info[files+'_fil'][:,1] += i
#        test.info[files+'_over'][:,1] += i
#        i+=0.2
#        aa.append(files)
#        aa.append(files+'_fil')
#        aa.append(files+'_over')
#
#aa.append('result.txt')
#test.all_output(aa)


#i = 0.0
#aa = []
#bb = []       
#for files in os.listdir():
#    if files == 'input.txt':
#        continue
#    if not '.s' in files:
#        test.read_input(files,files)
#        #loc = test.info[files][:,1].argmax(0)
#        #loc = np.min(np.where(np.absolute(test.info[files][:,0]-30)<0.05)) #0.05, 0.01...
#        #test.info[files][:,1] /= test.info[files][loc,1]
#        test.low_pass_filter(files,files+'_fil')
#        #test.overlap_filter(files+'_fil',files+'_over',529, 'end', 10)
#        loc = np.min(np.where(np.absolute(test.info[files][:,0]-540)<0.1))
#        #loc = np.min(np.where(np.absolute(test.info[files][:,0]-35)<0.05)) #0.05, 0.01...
#        norm = test.info[files][loc,1]
#        test.info[files+'_origin'] = copy.deepcopy(test.info[files][:,:])
#        test.info[files+'_filtered'] = copy.deepcopy(test.info[files+'_fil'][:,:])
#        test.info[files][:,1] /= norm
#        test.info[files+'_fil'][:,1] /= norm
#        #test.info[files+'_over'][:,1] /= norm
#
#        test.info[files][:,1] += i
#        test.info[files+'_fil'][:,1] += i
#        #test.info[files+'_over'][:,1] += i
#        #i+=0.6
#        bb.append(files+'_origin')
#        bb.append(files+'_filtered')
#        aa.append(files)
#        aa.append(files+'_fil')
#        #aa.append(files+'_over')
#
#aa.append('result.dat')
#bb.append('resutl1.dat')
#test.all_output(aa)
#test.all_output(bb)

def autowriting(dataset,typ,normloc,ovl,ovstart,ovend,ovch,normstep):
    i = .0
    origin_out = []
    normed_out = []
    for files in os.listdir():
        if files == 'input.txt':
            continue
        if typ in files:
            name = os.path.splitext(files)[0]

            dataset.read_input(name,files)
            dispersion = dataset.info[dataset.name][1,0] - dataset.info[dataset.name][0,0]

            dataset.low_pass_filter(name,name+'_Low-Pass')
            if(ovl == True):
                dataset.overlap_filter(name+'_Low-Pass',name+'_Over', ovstart, ovend, ovch)

            loc = np.min(np.where(np.absolute(dataset.info[name][:,0]-normloc)<dispersion))
            norm = dataset.info[name][loc,1]
            dataset.info[name+' '] = copy.deepcopy(dataset.info[name][:,:])
            dataset.info[name+'_Low-Pass '] = copy.deepcopy(dataset.info[name+'_Low-Pass'][:,:])
            if(ovl == True):
                dataset.info[name+'_Over '] = copy.deepcopy(dataset.info[name+'_Over'][:,:])

            dataset.info[name][:,1] /= norm
            dataset.info[name+'_Low-Pass'][:,1] /= norm
            if(ovl == True):
                dataset.info[name+'_Over'][:,1] /= norm
                dataset.info[name+'_Over'][:,1] += i
            dataset.info[name][:,1] += i
            dataset.info[name+'_Low-Pass'][:,1] += i

            origin_out.append(name+' ')
            origin_out.append(name+'_Low-Pass ')
            if(ovl == True):
                origin_out.append(name+'_Over ')

            normed_out.append(name)
            normed_out.append(name+'_Low-Pass')
            if(ovl == True):
                normed_out.append(name+'_Over')
            
            i += normstep

    origin_out.append('Original_Data.dat')
    normed_out.append('Normalized_Data.dat')
    dataset.all_output(origin_out)
    dataset.all_output(normed_out)


#dataset,typ,normloc,ovl,ovstart,ovend,ovch,normstep
#ex: autowriting(data,'ep1',5,True,2,'end',10,0.2)
#autowriting(data,'loss',18,True,2.5,10,5,0.2)

#data_out = []
#typ = 'ep2'
#for files in os.listdir():
#    if files == 'input.txt':
#        continue
#
#    if typ in files:
#        name = os.path.splitext(files)[0]
#        data.read_input(name,files)
#        
#        #loc_start = np.min(np.where(np.absolute(data.info[name][:,1])>0.1))
#        #start = data.info[name][loc_start,0]
#        data.smoothing_spline(name,name+'_smoothe','start',1,0.99995)
#        data.smoothing_spline(name+'_smoothe',name+'_smoothed',1,'end',0.9999999)
#        #data.smoothing_spline(name,name+'_smoothe',start,20,0.999999)
#        #data.smoothing_spline(name+'_smoothe',name+'_smoothed',20,'end',0.999999)
#        data_out.append(name)
#        data_out.append(name+'_smoothed')
#
#        #xtemp = copy.deepcopy(data.info[name+'_smoothed'][:,0])
#        #ytemp = copy.deepcopy(data.info[name+'_smoothed'][:,1])
#        #data.info[name+'_cond'] = np.inner(xtemp[:,0], ytemp[:,0])
#        #data.info[name+'_joint'] = np.inner(xtemp[:,0], data.info[name+'_cond'][:,0])
#        #data_out.append(name+'_cond')
#        #data_out.append(name+'_joint')
#    
#
#data_out.append('output.dat')
#data.all_output(data_out)

"""Example Usage"""

"""
#path = os.path.dirname(os.path.realpath(__file__))
#os.chdir(os.path.join(path, 'input'))
#data = Spectrum()

data_out = []
typ = 'test'
for files in os.listdir():
    if files == 'input.txt':
        continue
    if typ in files:
        name = os.path.splitext(files)[0]
        data.read_input(name,files)
        
        data.smoothing_spline(name,name+'_smoothe','start',530,0.9995)
        data.smoothing_spline(name+'_smoothe',name+'_smoothed',530,'end',0.99995)
        data_out.append(name)
        data_out.append(name+'_smoothed')
        
data_out.append('output.dat')
data.all_output(data_out)
"""

if __name__ == "__main__":
    """
    This section will be executed only if python directly calls it, i.e. python EELS.py
    1) https://stackoverflow.com/questions/419163/what-does-if-name-main-do
    2) https://www.bouncybouncy.net/blog/how-not-to-program-in-python/
    """
    data = Spectrum()
    data.set_dir(os.getcwd())
    
    
    """
    When the interpreter works, sys.path list is initialized.
    sys.path contains the module search paths via PYTHONPATH and possibly via .pth files
    
    Though PYTHONPATH and .pth files live in the operating system, sys.path settings endure
    for only as long as the Python session or program tha made them runs; they are not retained
    after Python exits.
    
    How to find my PYTHONPATH:
        import os
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    
    References:
        1) Learning Python: Powerful Object-Oriented Programming By Mark Lutz
        2) https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    """
    sys.path.append('C:\\Users\\Subin Bang\\Dropbox\\EELS Data\\EELS_Filter')
    #from EELS import Spectrum