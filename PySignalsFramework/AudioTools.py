"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalFramework
File:           AudioTools.py
Description:
"""

                #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
import scipy.io.wavfile

                #### CLASS DEFINITIONS ####

class AudioIO :
    """
    Static Class of methods to Input/Output Audio
    --------------------------------
    [No member variables for static class]
    --------------------------------
    Make No Instance
    """

    def __init__(self):
        """ False Constructor for Plotting Static Class - Raises Error """
        raise TypeError("Type 'AudioIO' is a static class - cannot make instance")

    @staticmethod
    def readWAV(localPath,channels="all"):
        """ Read wav Audio file and return sample rate and audio data """
        rate,data = scipy.io.wavfile.read(localPath)
        if channels.capitalize() in ["All","BOTH","LR"]:
            return rate,data.transpose()
        elif channels.capitalize() in ["LEFT","L"]:
            return rate,data.transpose()[0]
        elif channels.capitalize() in ["Right","R"]:
            return rate,data.transpose()[1]
        else:
            raise ValueError("Channels keyword must be in [left,right,all]")

    @staticmethod
    def readTXT(localPath,cols,sampleRate=44100):
        """ Read txt Audio file and return sampl rate and audio data """
        raise NotImplementedError()

    @staticmethod
    def writeWAV(localPath,signal,sampleRate):
        """ Write wav Audio file to local path """
        try:
            #signal = signal.astype(np.float32)
            scipy.io.wavfile.write(localPath,sampleRate,signal)
            return True
        except Exception as expt:
            print(expt)
            return False
        
    @staticmethod
    def writeTXT(localPath,signal):
        """ Write txt Audio file to local path """
        raise NotImplementedError()

class AudioSamples:
    """
    Load in Locally stored audio samples
    --------------------------------
    [No member variables for static class]
    --------------------------------
    Make No Instance
    """

    def __init__(self):
        """ False Constructor for Plotting Static Class - Raises Error """
        raise TypeError("Type 'AudioSamples' is a static class - cannot make instance")

class Signal:
    """
    Signal Data Type
        Holds Data for time-series or frequency-series data
    --------------------------------
    _data           arr[float]          Array to hold signal data
    _domain         str                 Indicate if signal is in time/freq domain
    _sampleRate     int                 Sample rate of data in the signal
    --------------------------------

    """

    def __init__(self,data=None,shape=None,dType=float,domain='time',sampleRate=44100):
        """ Constructor for Signal Instance """
        self._data = None
        self._domain = None
        self._sampleRate = None
        self._frameParams = None

        self.setData(data)
        self.setDomain(domain)
        self.setSampleRate(sampleRate)
        self.setShape(shape)

    def __del__(self):
        """ Destructor for Signal Instance """
        pass

    def deepCopy(self):
        """ Copy Constructor for Signal Instance """
        result = Signal(np.copy(self._data),self._domain,self._sampleRate)
        return result

    """ Getters and Setters """

    def getData(self):
        """ Get the raw Signal Array """
        return self._data

    def setData(self,x):
        """ Set the raw Signal Array """
        if (x is None):
            self._data = x
        elif (type(x) != np.ndarray):
            x = np.array(x)
        self._data = x
        return self
    
    def getDomain(self):
        """ Get if Signal is in Time or Freq Domain """
        return self._domain

    def setDomain(self,x):
        """ Set if Signal is in TIme of Freq Domain """
        if (x.upper() not in ["TIME","FREQ"]):
            raise ValueError("Domain must be a string in ['TIME','FREQ','BOTH']")
        self._domain = x.upper()
        return self

    def getSampleRate(self):
        """ Get the Sample Rate """
        return self._sampleRate
        
    def setSampleRate(self,x):
        """ Set the Sample Rate """
        if (x <= 0):
            raise ValueError("Sample Rate must be Greater than or equal to Zero")
        else:
            self._sampleRate = x
        return self

    def getShape(self):
        """ Get the Shape of the Signal """
        if (self._data):
            return self._data.shape
        else:
            return (None,)

    def setShape(self,x):
        """ Set the Shape of the Signal """
        if (x is None):             # No shape Given, ignore
            return self
        if (len(x) > 2):            # more than 2D
            raise ValueError("Signal must be 1 or 2 dim, but got " + str(len(x)))
        if (self._data is None):
            self._data = np.empty(shape=x)
        else:
            self._data.reshape(x)
        return self

    def getDataType(self):
        """ Get the Data Type of the Signal """
        if (self._data is None):        # No signal yet
            return None
        else:
            return self._data.dtype

    def setDataType(self,x):
        """ Set the Data Type of the Signal """
        try:
            self._data = self._data.astype(dtype=x)
            return True
        except Exception as ecxpt:
            print(ecxpt)
        return False

    """ Public Interface """

    def getTimeAxis(self):
        """ Get the Time Axis for the Signal """
        return None

    def getFreqAxis(self):
        """ Get the Frequency Axis for the Signal """
        return None

    def copyDataToSignal(self,otherSignal):
        """ Create A Deep Copy of this Signal's Data """
        try:
            self.setData(np.copy(otherSignal.getData()))
            return True
        except Exception as ecxpt:
            print(ecxpt)
        return False

    def readFromWav(filePath):
        """ Read Audio file and return as Signal Instance """
        sampleRate,signal = AudioIO.readWAV(filePath)
        self.setSampleRate(sampleRate)
        self.setData(signal)
        return result

    def writeToWav(filePath):
        """ Write current Signal to time-series audio file"""
        if (self._domian != "TIME"):
            print("WARNING - Attempting to export non-time-series signal!")
        try:
            AudioIO.writeWav(filePath,self._data,self._sampleRate)
            return True
        except Exception as excpt:
            print(excpt)
            return False
    

    """ Private Interface """

    

    """ Magic Methods  and Static Methods """

    def __str__(self):
        """ Return String Representation of Instance """
        result = ""
        result += self.getDomain() + " signal"
        return result

    def __repr__(self):
        """ Return Programmer String representation of Instance """
        result = "PySignalsFramework.AudioTools.Signals object"
        result += "shape: " + [str(x) for x in self._data.shape]
        return result

    def __getitem__(self,idx):
        """ Overload Index Operator """
        return self._data[idx]

    def __iter__(self):
        """ Define Forward Iterator for Signal Instance """
        for val in self._data:
            yield val

    def __add__(self,x):
        """ Overload Addition Operator """
        return self._data + x

    def __sub__(self,x):
        """ Overload Subtraction Operator """
        return self._data - x

   


class FrameParams:
    """
    FrameParams Type - Class to hold data related to constructing and destructing analysis Frmaes
    --------------------------------
    _samplesPerFrame        int         Number of waveform samples in each analysis Frames
    _samplesOverlap         int         Number of samples overlap between each frame
    _maxNumFrames           int         Maximum number of frames in matrix
    _framesInUse            int         Current number if frames in matrix
    _padTail                int         Number of zeros to tail-pad each analysis frame
    _padHead                int         Number of zeros to head-pad each analysis frame
    --------------------------------
    """

    def __init__(self,samplesPerFrame,samplesOverlap,maxFrames,tailPad,headPad):
        """ Constructor for FrameParams Instance """
        self._samplesPerFrame       = samplesPerFrame
        self._samplesOverlap        = samplesOverlap
        self._maxNumFrames          = maxFrames
        self._framesInUse           = 0
        self._padTail               = padTail
        self._padHead               = padHead

    def __del__(self):
        """ Destructor for FrameParams Instance """
        pass

    def deepCopy(self):
        """ Create a New FramesParams Instance as a deep copy from this one """
        result = FrameParams(self._samplesPerFrame,self._samplesOverlap,
                             self._maxNumFrames,self._padTail,self._padHead)
        return result

    """ Getters and Setters """

    def getSamplesPerFrame(self):
        """ Get Number of Samples per Frame """
        return self._samplesPerFrame

    def setSamplesPerFrame(self,x):
        """ Set Number of Samples per Frame """
        self._samplesPerFrame = x
        return self

    def getSamplesOverlap(self):
        """ Get Number of Samples overlap """
        return self._samplesOverlap

    def setSamplesOverlap(self,x):
        """ Set Number of Samples Overlap """
        self._samplesOverlap = x
        return self

    def getMaxNumFrames(self):
        """ Get Maximum Number of Analysis Frames """
        return self._maxNumFrames

    def setMaxNumFrames(self,x):
        """ Set the Maximum Number of Analysis Frames """
        self.__maxNumFrames = x
        return self

    @property
    def framesInUse(self):
        """ Return Refrence to number of frames in use """
        return self._framesInUse

    def getPadding(self):
        """ Get Head/Tail Padding """
        return (self._padHead,self._padTail,)

    def setPadding(self,head=None,tail=None):
        if (head is not None):
            self._padHead = head
        if (tail is not None):
            self._padTail = tail
        return self

    def getPercentOverlap(self):
        """ Get Percent Overlap Between frames """
        return (self._samplesOverlap / self._samplesPerFrame)

    def getFrameSize(self):
        """ Get total size of Each Frame """
        return (self._padHead + self._samplesPerFrame + self._padTail)

    """ Magic Methods """

    def __str__(self):
        """ Get string representation of instance """
        result = ""
        return result


class WavesGenerator :
    """
    SimpleWaves Type - Methods to create signals of any length and component frequencies 
    --------------------------------
    _time           arr[int]        1 x N shaped array
    _nVals          int             Number of frequencies in signal
    _freq           arr[float]      1 x M array of linear - frequency values
    _amps           arr[float]      1 x M array of amplitude coefficients
    _phas           arr[float]      1 x M array of phase shifts
    _sampleRate     int             Sample rate for this audio
    --------------------------------
    """

    def __init__(self,time,linearFrequencies=[],amplitudes=None,phases=None,sampleRate=44100):
        """ Constructor for SimpleWaves Instance """
        self._time = time
        self._freq = np.array(linearFrequencies).ravel()
        self._nVals = self._freq.shape[-1]
        self._amps = self.setWaveformParams(amplitudes)
        self._phas = self.setWaveformParams(phases) * 0
        self._sampleRate = sampleRate
        
    def setWaveformParams (self,value):
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

    def validateWaveformParams(self):
        """ test if all params have correct shape """
        self._nVals = self._freq.shape[-1]
        if (self._amps.shape[-1] != self._nVals) or (self._phas.shape[-1] != self._nVals):
            errmsg = "frequency, amplitude, and phase arrays must all have the same shape\n"
            errmsg += "Instead, found: {}, {}, {}".format(self._freq,self._amps,self._phas)
            raise ValueError(errmsg)
        else:
            return True
    
    def getCosineWave(self):
        """ Create Cosine wave given class attributes """
        self.validateWaveformParams()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A*np.cos(2*np.pi*f*self._time + p)
        return signal

    def getSineWave(self):
        """ Create Sine wave given class attributes """
        self.validateWaveformParams()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A*np.sin(2*np.pi*f*self._time + p)
        return signal

    def getSquareWave(self):
        """ Create Square wave given class attributes """
        self.validateWaveformParams()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A * scisig.square(2*np.pi*f*self._time + p)
        return signal

    def getSawtoothWave(self):
        """ Create Sawtooth wave given class attributes """
        self.validateWaveformParams()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A * scisig.sawtooth(2*np.pi*f*self._time + p)
        return signal

    def getTriangleWave(self):
        """ Create Triangle wave given class attributes """
        self.validateWaveformParams()
        signal = np.zeros(shape=self._time.shape[-1])
        for A,f,p in zip(self._amps,self._freq,self._phas):
            signal += A * scisig.sawtooth(2*np.pi*f*self._time + p,width=0.5)
        return signal

    def getAmplitudes(self):
        """ Get Amplitude values """
        return self._amps

    def getFrequencies(self):
        """ Get Linear Frequency values """
        return self._freq

    def getPhases(self):
        """ Get Phase Shift values """
        return self._phas

class SimpleWavesGenerator:
    """
    Static Class of methods to Generate Simple Waveforms
    --------------------------------
    [No member variables for static class]
    --------------------------------
    Make No Instance
    """

    def __init__(self):
        """ False Constructor for Plotting Static Class - Raises Error """
        raise TypeError("Type 'SimpleWavesGenerator' is a static class - cannot make instance")

    @staticmethod
    def CosineWave(time,amp=1,freq=1,phase=0):
        """ Create Cosine wave given amplitude, linear frequency, time axis, and phase shift """
        signal = amp*np.cos(2*np.pi*freq*time + phase)
        return signal

    @staticmethod
    def SineWave(time,amp=1,freq=1,phase=0):
        """ Create Sine wave given amplitude, linear frequency, time axis, and phase shift """
        signal = amp*np.sin(2*np.pi*freq*time + phase)
        return signal

    @staticmethod
    def SquareWave(time,amp=1,freq=1,phase=0):
        """ Create Square wave given amplitude, linear frequency, time axis, and phase shift """
        signal = amp*scisig.square(2*np.pi*freq*time + phase)
        return signal

    @staticmethod
    def SawtoothWave(time,amp=1,freq=1,phase=0):
        """ Create Sawtooth wave given amplitude, linear frequency, time axis, and phase shift """
        signal = amp*scisig.sawtooth(2*np.pi*freq*time + phase)
        return signal

    @staticmethod
    def TriangleWave(time,amp=1,freq=1,phase=0):
        """ Create Triangle wave given amplitude, linear frequency, time axis, and phase shift """
        signal = scisig.sawtooth(2*np.pi*freq*time + phase,width=0.5)
        return signal


class WindowFunctions :
    """
    static class of window functions
    --------------------------------
    [No member variables for static class]
    --------------------------------
    Make No Instance
    """

    def __init__(self):
        """ False Constructor for Plotting Static Class - Raises Error """
        raise TypeError("Type 'Plotting' is a static class - cannot make instance")

    @staticmethod
    def windowFunctions(functionName,nSamples):
        """ Get A window function from string identifying it """
        windows = {"blackman":scisig.windows.blackman,}
        raise NotImplementedError()

    @staticmethod
    def hanningWindow(nSamples):
        """ Get Hanning Window that is nSamples Long """
        return scisig.windows.hann(nSamples)

    @staticmethod
    def gaussianWindow(nSamples):
        """ Get Hanning Window that is nSamples Long """
        return scisig.windows.gaussian(nSamples)


class Plotting:
    """
    Class of Static methods to provide matplotlib visualizations of data
    --------------------------------
    [No member variables for static class]
    --------------------------------
    Make No Instance
    """

    def __init__(self):
        """ False Constructor for Plotting Static Class - Raises Error """
        raise TypeError("Type 'Plotting' is a static class - cannot make instance")

    @staticmethod
    def plotTimeSeries(xData,yData,labels=[],title="",save=False,show=True):
        """ Plot Time-series information in Matplotlib figure """
        plt.figure(figsize=(16,12))
        plt.title(label=title,size=40,weight='bold')
        plt.xlabel("Time [samples]",size=20,weight='bold')
        plt.ylabel("Amplitude [units]",size=20,weight='bold')

        # Each are 1D arrays
        if (xData.ndim == 1) and (yData.ndim == 1):
            plt.plot(xData,yData,color='blue',label=labels)
            plt.hlines(0,min(xData),max(xData),color='black')
            plt.vlines(0,min(yData),max(yData),color='black')

        # 1D and 2D arrays
        elif (xData.ndim == 1) and (yData.ndim == 2):
            plt.plot(xData,yData,label=labels)
            plt.hlines(0,np.min(xData),np.max(xData),color='black')
            plt.vlines(0,np.min(yData),np.max(yData),color='black')

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
    def plotFrequencySeries(xData,yData,labels=[],title="",save=False,show=True):
        """ Plot Frequency-series information in Matplotlib figure """
        return None

    @staticmethod
    def plotGeneric(xData,yData,labels=[],title="",save=False,show=True):
        """ Plot Generic Information in Matplotlib Figure """

        # Init Figures        
        plt.figure(figsize=(16,12),facecolor='gray')
        plt.title(title,size=40,weight='bold')
        plt.xlabel("X - Axis",size=30,weight='bold')
        plt.ylabel("Y - Axis",size=30,weight='bold')

        # Each are 1D arrays
        if (xData.ndim == 1) and (yData.ndim == 1):
            plt.plot(xData,yData,color='blue',label=labels)
            plt.hlines(0,min(xData),max(xData),color='black')
            plt.vlines(0,min(yData),max(yData),color='black')

        # 1D and 2D arrays
        elif (xData.ndim == 1) and (yData.ndim == 2):
            plt.plot(xData,yData,label=labels)
            plt.hlines(0,np.min(xData),np.max(xData),color='black')
            plt.vlines(0,np.min(yData),np.max(yData),color='black')

        plt.grid()
        plt.legend()
        
        if save :         
            if type(save) == str:
                plt.savefig(save + ".png")
            else:
                plt.savefig(title + ".png")
        if show == True:
            plt.show()

        plt.close()
        return None