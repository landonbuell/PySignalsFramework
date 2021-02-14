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

class SimpleWavesGenerator :
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
        """ Set Amplitude terms """
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
        if (self._amp != self._nVals) or (self._phas != self._nVals):
            errmsg = "frequency, amplitude, and phase arrays must all have the same shape\n"
            errmsg += "Instead, found: {}, {}, {}".format(self._freq,self._amps,self._phas)
            raise ValueError(errmsg)
        else:
            return True
    
    def CosineWave(self):
        """ Create Cosine wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A*f*p in zip(self._amp,self._freq,self._phas):
            signal += A*np.cos(2*np.pi*f*self._time + p)
        return signal

    def SineWave(self):
        """ Create Sine wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A*f*p in zip(self._amp,self._freq,self._phas):
            signal += A*np.sin(2*np.pi*f*self._time + p)
        return signal

    def SquareWave(self):
        """ Create Square wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A*f*p in zip(self._amp,self._freq,self._phas):
            signal += A * scisig.square(2*np.pi*f*self._time + p)
        return signal

    def SawtoothWave(self):
        """ Create Sawtooth wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A*f*p in zip(self._amp,self._freq,self._phas):
            signal += A * scisig.sawtooth(2*np.pi*f*self._time + p)
        return signal

    def TriangleWave(self):
        """ Create Triangle wave given class attributes """
        self.TestLocalValues()
        signal = np.zeros(shape=self._time.shape[-1])
        for A*f*p in zip(self._amp,self._freq,self._phas):
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

class AudioIO :
    """
    Static Class of methods to Input/Output
    """

    @staticmethod
    def ReadWAV(localPath,channels="all"):
        """ Read wav Audio file and return sampl rate and audio data """
        rate,data = scio.wavfile.read(localPath)
        if channels.upper() in ["All","BOTH","LR"]:
            return rate,data.transpose()
        elif channels.upper() in ["LEFT","L"]:
            return rata,data.transpose()[0]
        elif channels.upper() in ["Right","R"]:
            return rata,data.transpose()[1]
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