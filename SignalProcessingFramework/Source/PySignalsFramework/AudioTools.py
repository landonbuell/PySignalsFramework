"""
Landon Buell
EffectsEmmulatorPython
Toy Audio Samples
5 Feb 2020
"""

                #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
import scipy.io as scio

                #### CLASS DEFINITIONS ####

class WavesGenerator :
    """
    SimpleWaves Type - Methods to create signals of any length and component frequencies 
    --------------------------------
    _time (arr[int]) : 1 x N shaped array
    _nVals (int) : Number of frequencies in signal
    _freq (arr[float]) : 1 x M array of linear - frequency values
    _amps (arr[float]) : 1 x M array of amplitude coefficients
    _phas (arr[float]) : 1 x M array of phase shifts
    _sampleRate (int) : Sample rate for this audio
    --------------------------------
    """

    def __init__(self,time,linearFrequencies=[],amplitudes=None,phases=None,sampleRate=44100):
        """ Constructor for SimpleWaves Instance """
        self._time = time
        self._freq = np.array(linearFrequencies).ravel()
        self._nVals = self._freq.shape[-1]
        self._amps = self.SetLocalValues(amplitudes)
        self._phas = self.SetLocalValues(phases) * 0
        self._sampleRate = sampleRate
        
    def SetLocalValues (self,value):
        """ Set Local value terms based on data type """
        if value:
            if type(values) == int:
                values = np.ones(shape=self._nVals,dtype=np.float32)*value
            elif type(value) == list or type(value) == np.ndarray:
                if len(value) != self._nVals:
                    raise ValueError("Ampltidues must be int or list w/ same shape as self._freq")
                values = np.array(value).ravel()
            else:
                raise TypeError("Argument must be of type int or list[int]")
        else:
            values = np.ones(shape=self._nVals,dtype=np.float32)
        return values

    def TestLocalValues(self):
        """ test if all params have correct shape """
        self._nVals = self._freq.shape[-1]
        if (self._amps.shape[-1] != self._nVals) or (self._phas.shape[-1] != self._nVals):
            errmsg = "frequency, amplitude, and phase arrays must all have the same shape\n"
            errmsg += "Instead, found: {}, {}, {}".format(self._freq,self._amps,self._phas)
            raise ValueError(errmsg)
        else:
            return True
    
    def CosineWave(self):
        """ Create Cosine wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A*np.cos(2*np.pi*f*self._time + p)
        return signal

    def SineWave(self):
        """ Create Sine wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A*np.sin(2*np.pi*f*self._time + p)
        return signal

    def SquareWave(self):
        """ Create Square wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A * scisig.square(2*np.pi*f*self._time + p)
        return signal

    def SawtoothWave(self):
        """ Create Sawtooth wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A * scisig.sawtooth(2*np.pi*f*self._time + p)
        return signal

    def TriangleWave(self):
        """ Create Triangle wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A * scisig.sawtooth(2*np.pi*f*self._time + p,width=0.5)
        return signal

    @property
    def GetAmplitudes(self):
        """ Get Amplitude values """
        return self._amps

    @property
    def GetFrequencies(self):
        """ Get Linear Frequency values """
        return self._freq

    @property
    def GetPhases(self):
        """ Get Phase Shift values """
        return self._phas

class SimpleWavesGenerator:
    """
    Static Class of methods to Generate Simple Waveforms
    """

    @staticmethod
    def CosineWave(amp,freq,time,phase):
        """ Create Cosine wave given class attributes """
        signal = amp*np.cos(2*np.pi*freq*time + phase)
        return signal

    @staticmethod
    def SineWave(amp,freq,time,phase):
        """ Create Sine wave given class attributes """
        signal = amp*np.sin(2*np.pi*freq*time + phase)
        return signal

    @staticmethod
    def SquareWave(amp,freq,time,phase):
        """ Create Square wave given class attributes """
        signal = amp*scisig.square(2*np.pi*freq*time + phase)
        return signal

    @staticmethod
    def SawtoothWave(amp,freq,time,phase):
        """ Create Sawtooth wave given class attributes """
        signal = amp*scisig.sawtooth(2*np.pi*freq*time + phase)
        return signal

    @staticmethod
    def TriangleWave(amp,freq,time,phase):
        """ Create Triangle wave given class attributes """
        signal = scisig.sawtooth(2*np.pi*freq*time + phase,width=0.5)
        return signal

class AudioIO :
    """
    Static Class of methods to Input/Output
    """

    @staticmethod
    def ReadWAV(localPath,channels="all"):
        """ Read wav Audio file and return sample rate and audio data """
        rate,data = scio.wavfile.read(localPath)
        if channels.upper() in ["All","BOTH","LR"]:
            return rate,data.transpose()
        elif channels.upper() in ["LEFT","L"]:
            return rate,data.transpose()[0]
        elif channels.upper() in ["Right","R"]:
            return rate,data.transpose()[1]
        else:
            raise ValueError("Channels keyword must be in [left,right,all]")

    @staticmethod
    def ReadTXT(localPath,cols,sampleRate=44100):
        """ Read txt Audio file and return sampl rate and audio data """
        raise NotImplementedError()

    @staticmethod
    def WriteWAV(localPath,signal,sampleRate):
        """ Write wav Audio file to local path """
        raise NotImplementedError()

    @staticmethod
    def WriteTXT(localPath,signal):
        """ Write txt Audio file to local path """
        raise NotImplementedError()


class WindowFunctions :
    """
    static class of window functions
    """

    @staticmethod
    def WindowFunctions(functionName,nSamples):
        """ Get A window function from string identifying it """
        windows = {"blackman":scisig.windows.blackman,}
        raise NotImplementedError()

    @staticmethod
    def HanningWindow(nSamples):
        """ Get Hanning Window that is nSamples Long """
        return scisig.windows.hann(nSamples)

    @staticmethod
    def GaussianWindow(nSamples):
        """ Get Hanning Window that is nSamples Long """
        return scisig.windows.gaussian(nSamples)


class Plotting:
    """
    Class of Static methods to provide matplotlib visualizations of data
    """

    @staticmethod
    def PlotTimeSeries(xData,yData,labels=[],title="",save=False,show=True):
        """
        Plot Time-series information in Matplotlib figure
        --------------------------------
        xData (list/arr) : n-Dim array of values to plot as x-axis values
        yData (list/arr) : n-Dim array of values to plot as y-axis values
        labels (list) : List of strings to use as column lables
        title (str) : String of text to use as plot title / save name
        save (bool) : If true, save the figure to the current working directory
        show (bool) : If true, plot the figure to console
        --------------------------------
        Optionally save/show plot
        """
        plt.figure(figsize=(16,12))
        plt.title(label=title,size=40,weight='bold')
        plt.xlabel("Time [samples]",size=20,weight='bold')
        plt.ylabel("Amplitude [units]",size=20,weight='bold')

        # Each are 1D arrays
        if (xData.ndim == 1) and (yData.ndim == 1):
            plt.plot(xData,yData,color='blue',label=labels)
            plt.hlines(0,min(xData),max(xData),color='black')
            plt.vlines(0,min(yData),max(xData),color='black')

        plt.grid()
        plt.legend()

        if save == True:
            saveName = title.replace(" ","")
            plt.savefig(saveName+".png")
        if show == True:
            plt.show()
        plt.close()
        return None
        

    @staticmethod
    def PlotFrequencySeries(xData,yData,labels=[],title="",save=False,show=True):
        """
        Plot Time-series information in Matplotlib figure
        --------------------------------
        xData (list/arr) : n-Dim array of values to plot as x-axis values
        yData (list/arr) : n-Dim array of values to plot as y-axis values
        labels (list) : List of strings to use as column lables
        title (str) : String of text to use as plot title / save name
        save (bool) : If true, save the figure to the current working directory
        show (bool) : If true, plot the figure to console
        --------------------------------
        Optionally save/show plot
        """
        return None

    @staticmethod
    def PlotGeneric(xData,yData,labels=[],title="",save=False,show=True):
        """
        Plot Time-series information in Matplotlib figure
        --------------------------------
        xData (list/arr) : n-Dim array of values to plot as x-axis values
        yData (list/arr) : n-Dim array of values to plot as y-axis values
        labels (list) : List of strings to use as column lables
        title (str) : String of text to use as plot title / save name
        save (bool) : If true, save the figure to the current working directory
        show (bool) : If true, plot the figure to console
        --------------------------------
        Optionally save/show plot
        """
        raise NotImplementedError()
        return None