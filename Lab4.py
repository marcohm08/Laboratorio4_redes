# Laboratorio 4 Redes de Computadores
# Marco Hernandez
# 19.318.862-1

import scipy
import scipy.io
import scipy.misc
from scipy import signal
from scipy import interpolate
from scipy import integrate
from scipy.io import wavfile
from scipy.fftpack import fft,fftshift,fftfreq,ifft

import numpy as np
import matplotlib.pyplot as plt
import random

import copy

import sys
from urllib3.connectionpool import xrange




A = 2 # amplitud del coseno

# function to generate signal when a bit of the digital signal is 1
# inputs:
    # t : time value 
    # fc: frecuency of the cos function
# output: cos function value in t
def cosFunction(t,fc):
    return A*np.cos(2*np.pi*fc*t)

# function to calculate number of diferent bits of a demodulated digital signal in comparison to the original signal
# inputs:
    # original: original signal
    # demodulated: demodulated signal
def calculateBER(original, demodulated):
    diferent = 0
    for i in xrange(0,len(original)):
        if(original[i] != demodulated[i]):
            diferent+=1
    ber = diferent/len(original)
    return ber 

# Class to represent a digital signal
class DigitalSignal:
    def __init__(self,bps): 
        self.signal = []# signal array
        self.signalDemod = []# signal after modulate
        self.time = []# time array
        self.modTime = []# time array to the modulated signal
        self.bps = bps # bits per second
        self.bitTime = 1/bps# bit time
    # method to generate a digital signal randomly with a previous seed determined by the user
    # inputs: 
        # lenght: number ob bits to create the digital signal
        # seed: seed to the random numbers
    def randomSignal(self,lenght,seed):
        init = self.bitTime
        random.seed(seed)
        for i in range(0,lenght):
            self.signal.append(random.randrange(0,2))
            self.time.append(init)
            init= init + self.bitTime
    def setBps(self,bps):
        self.bitTime = 1/bps
        init = self.bitTime
        self.time = []
        for i in range(0, len(self.signal)):
            self.time.append(init)
            init= init + self.bitTime

    # method to modulate by frequency a digital signal
        #fc : frecquency to apply to the cos
    def ookMod(self,fc):
        t = [] 
        init = 0 
        i = 0
        modulated = []
        self.sampleStep = self.bitTime/(2*fc)
        for bit in self.signal: 
            bitSignal = []    
            tn = np.linspace(init,self.time[i], 2*fc)
        
            if (bit == 0): 
                for t_i in tn: 
                    value = 0.0 # when the bit is 0, the value is also 0
                    bitSignal.append(value) 
                    t.append(t_i) 
            elif (bit == 1): 
                for t_i in tn:
                    value = cosFunction(t_i,fc) # when the bit is 1, the value is the cos function in the time
                    bitSignal.append(value)
                    t.append(t_i)   
            modulated.append(bitSignal)  
            init = self.time[i]
            i+=1
        self.modTime = t
        self.signalModulated = np.array(modulated)
    
    # Method to plot a grapf using matplotlib
    # Input
    #   number: number of plot
    #   xData: data of the x axis
    #   yData: data oj the y axis
    #   xlabel: name of the x axis
    #   ylabel: name of the y axis
    #   titlte: title of the plot
    def plotGraph(self,number, xData, yData, xLabel = None, yLabel = None, title = None):
        plt.figure(number)
        plt.plot(xData,yData)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
    
    # method to demodulate a ook modulated signal
        # inputs:
            # timeArray: time array of the signal modulated
            # signalArray: array of the signal to demodulate
        # output: array of signal demodulated
    def demodulate(self,timeArray,signalArray):
        maxValue = 0
        i = 0
        signalDemodulated = []
        for bitSignal in signalArray:
            maxValue = np.max(bitSignal)
            if maxValue >= A:
                signalDemodulated.append(1)
            else:
                signalDemodulated.append(0)
        return signalDemodulated

    #method to add AWGN noise to a modulated signal
    # inputs:
        # snr_db: level of noise expresed in SNR in decibels
    #output: array of signal modulated with noise
    def awgnAdd(self,snr_db): 
        self.signalPower = np.array(self.signalModulated)**2
        self.signalPowerDb = 10 * np.log10(self.signalPower)
        sig_avg = abs(np.mean(self.signalPower))
        sig_avg_db = 10 * np.log10(sig_avg)
        noise_avg_db = sig_avg_db - snr_db
        noise_avg = 10 ** (noise_avg_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_avg), len(self.modTime))
        signalWithNoise = []
        i = 0
        for bitSignal in self.signalModulated:
            noiseBitSignal = []
            for value in bitSignal:
                noiseBitSignal.append(value + noise[i])
                i+=1
            signalWithNoise.append(noiseBitSignal)
        return np.array(signalWithNoise)


if __name__ == "__main__":

    exampleSignal = DigitalSignal(10)# 10 bps signal
    exampleSignal.randomSignal(6,5)# create the signal array with 6 bits, 9 is the seed
    exampleSignal.ookMod(100) # modulate with a frequency of 100 HZ when the bit is 1
    signal1d = exampleSignal.signalModulated.ravel() # modulated signal transformed to 1d array

    plt.figure(1)
    plt.plot(exampleSignal.modTime,signal1d)
    plt.title("Señal modulada")
    plt.xlabel("f(t)")
    plt.ylabel("tiempo")   

    noiseExample = exampleSignal.awgnAdd(10)# add noise to the previous signal with an snr of 10 
    demExample = exampleSignal.demodulate(exampleSignal.modTime,noiseExample) #demodulate signal with noise

    plt.figure(2)
    plt.plot(exampleSignal.modTime, noiseExample.flatten())
    plt.title('Signal with noise')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')


    original = copy.deepcopy(exampleSignal.signal)
    original.insert(0,exampleSignal.signal[0])# only did this because to compare the signal with the modulated one
    time = copy.deepcopy(exampleSignal.time)
    time.insert(0,0)# only did this because to compare the signal with the modulated one
    fig, axs = plt.subplots(2)
    fig.suptitle('Original vs Demodulada')
    axs[0].step(time, original)
    axs[1].step(exampleSignal.modTime,exampleSignal.signalModulated.flatten())

    fig2, axs2 = plt.subplots(2)
    fig2.suptitle('Original vs Demodulada')
    axs2[0].step(exampleSignal.time, exampleSignal.signal)
    axs2[1].step(exampleSignal.time,demExample)
    

    snr_levels = [1,1.2,2,2.5,3,4,5,5.5,6,7]# list with levels ob snr
    bps = [1,4,7] # list with diferent bps
    bersPerSignal = [] # list to add ber of each signal
    for b in bps: 
        bers = [] # list to add bers of a signal with a especific bps
        if b == bps[0]:
            firstSignal = DigitalSignal(b)
            firstSignal.randomSignal(10**5,3)
        else:
            firstSignal.setBps(b) # adjust the time array to the new bps of the signal
        
        firstSignal.ookMod(10) # modulate signal with 10 HZ

        for level in snr_levels: # for each snr level
            noiseSignal = firstSignal.awgnAdd(level) # add noise
            demSignal = firstSignal.demodulate(firstSignal.modTime,noiseSignal) # demodulate signal
            ber = calculateBER(firstSignal.signal,demSignal) # calculate ber
            bers.append(ber) # add ber to bers list
        bersPerSignal.append(bers) # add bers list to the bersPerSignal

    plt.figure(5)
    plt.plot(snr_levels,bersPerSignal[0],label = str(bps[0]))
    plt.plot(snr_levels,bersPerSignal[1],label = str(bps[1]))
    plt.plot(snr_levels,bersPerSignal[2],label = str(bps[2]))
    plt.title('BER vs SNR')
    plt.ylabel('BER')
    plt.xlabel('SNR')
    plt.legend() 

    plt.show()



