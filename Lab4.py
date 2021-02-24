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



n = 100 # bits por segundo
n_a = 10 # numero de bits del arreglo
A = 2 # amplitud del coseno
t_s = 1/n #tiempo símbolo

# Funcion coseno que se aplica en FSK
def cosFunction(t,fc):
    return A*np.cos(2*np.pi*fc*t)

def calculateBER(original, demodulated):
    diferent = 0
    for i in xrange(0,len(original)):
        if(original[i] != demodulated[i]):
            diferent+=1
    ber = diferent/len(original)
    return ber 

class DigitalSignal:
    def __init__(self,bps): 
        self.signal = []
        self.signalDemod = []
        self.time = []
        self.modTime = []
        self.bps = bps
        self.bitTime = 1/bps
    def randomSignal(self,lenght,seed):
        init = self.bitTime
        random.seed(seed)
        """ self.signal = [random.randrange(0,2) for n in range(0,lenght)]
        self.time = [init*n + init for n in range(0,lenght)] """
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

    def ookMod(self,fc):
        t = [] # arreglo de tiempo de la señal
        init = 0 # tiempo de inicio de la señal
        i = 0
        modulated = []
        self.sampleStep = self.bitTime/(2*fc)
        for bit in self.signal: # por cada bit de la señal digital
            bitSignal = []    
            tn = np.linspace(init,self.time[i], 2*fc)
        
            if (bit == 0): # si el bit es igual a 0
                for t_i in tn: # por cada elemento de arreglo de tiempo tn
                    value = 0.0 # se calcula el coseno en el tiempo t__i
                    bitSignal.append(value) # se agrega el valor a arreglo de valores modulados
                    t.append(t_i) # se agrga t_i al arreglo de tiempo der salida
            elif (bit == 1): # En caso de que el bit sea 1 es lo mismo quie en el caso anterior pero con 
                # la frecuencia cuando el bit es igual a 1
                for t_i in tn:
                    value = cosFunction(t_i,fc)
                    #print(value)
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
    def demodulate(self,timeArray,signalArray):
        #interpolatedF = interpolate.interp1d(self.modTime, self.signalModulated)
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

    exampleSignal = DigitalSignal(10)
    exampleSignal.randomSignal(6,4)
    exampleSignal.ookMod(100)
    signal1d = exampleSignal.signalModulated.ravel()

    plt.figure(1)
    plt.plot(exampleSignal.modTime,signal1d)
    plt.title("Señal modulada")
    plt.xlabel("f(t)")
    plt.ylabel("tiempo")   

    noiseExample = exampleSignal.awgnAdd(10)
    demExample = exampleSignal.demodulate(exampleSignal.modTime,noiseExample)

    plt.figure(2)
    plt.plot(exampleSignal.modTime, noiseExample.flatten())
    plt.title('Signal with noise')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')

    fig, axs = plt.subplots(2)
    fig.suptitle('Original vs Demodulada')
    axs[0].step(exampleSignal.time, exampleSignal.signal)
    axs[1].step(exampleSignal.time,demExample)
    



    snr_levels = [1,1.2,2,2.5,3,4,5,7,8,8.5]
    bps = [1,4,7]
    bersPerSignal = []
    for b in bps:
        bers = []
        if b == bps[0]:
            firstSignal = DigitalSignal(b)
            firstSignal.randomSignal(10**5,5)
        else:
            firstSignal.setBps(b)
        
        firstSignal.ookMod(10)

        for level in snr_levels:
            noiseSignal = firstSignal.awgnAdd(level)
            demSignal = firstSignal.demodulate(firstSignal.modTime,noiseSignal)
            ber = calculateBER(firstSignal.signal,demSignal)
            bers.append(ber)
            print(ber)
        bersPerSignal.append(bers)

    plt.figure(4)
    plt.plot(snr_levels,bersPerSignal[0],label = str(bps[0]))
    plt.plot(snr_levels,bersPerSignal[1],label = str(bps[1]))
    plt.plot(snr_levels,bersPerSignal[2],label = str(bps[2]))
    plt.title('BER vs SNR')
    plt.ylabel('BER')
    plt.xlabel('SNR')
    plt.legend()

    plt.show()



