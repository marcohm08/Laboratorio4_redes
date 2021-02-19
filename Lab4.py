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



n = 100 # bits por segundo
n_a = 10 # numero de bits del arreglo
A = 2 # amplitud del coseno
t_s = 1/n #tiempo símbolo

# Funcion coseno que se aplica en FSK
def cosFunction(t,fc):
    return A*np.cos(2*np.pi*fc*t)

def calculateBER(original, demodulated):
    diferent = 0
    for i in range(0,len(original)):
        if(original[i] != demodulated[i]):
            diferent+=1
    ber = diferent/len(original)
    return ber 

class DigitalSignal:
    def __init__(self,bps): 
        self.signal = []
        self.signalModulated = []
        self.signalDemod = []
        self.time = []
        self.modTime = []
        self.bps = bps
        self.bitTime = 1/bps
    def randomSignal(self,lenght):
        init = self.bitTime
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
        self.sampleStep = self.bitTime/(2*fc)
        for bit in self.signal: # por cada bit de la señal digital
            if(self.time[i] == self.time[-1]):
                if bit == 1:
                    tn = np.arange(init,self.time[i] + 2*self.sampleStep, self.sampleStep)
                if bit == 0:
                    tn = np.arange(init,self.time[i] + 2*self.sampleStep, 10)
            else:
                if bit == 1:     
                    tn = np.arange(init,self.time[i], self.sampleStep)
                if bit == 0:
                    tn = np.arange(init,self.time[i], 10)
            if (bit == 0): # si el bit es igual a 0
                for t_i in tn: # por cada elemento de arreglo de tiempo tn
                    value = 0 # se calcula el coseno en el tiempo t__i
                    self.signalModulated.append(value) # se agrega el valor a arreglo de valores modulados
                    t.append(t_i) # se agrga t_i al arreglo de tiempo der salida
            elif (bit == 1): # En caso de que el bit sea 1 es lo mismo quie en el caso anterior pero con 
                # la frecuencia cuando el bit es igual a 1
                for t_i in tn:
                    value = cosFunction(t_i,fc)
                    #print(value)
                    self.signalModulated.append(value)
                    t.append(t_i)     
            init = self.time[i]
            i+=1
        self.modTime = t
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
        for step in timeArray:
            if step in self.time or step == timeArray[-1]:
                #print(step)
                if (maxValue >= A):
                    signalDemodulated.append(1)
                else: 
                    signalDemodulated.append(0)
                maxValue = abs(signalArray[i])
            else:
                if(maxValue < signalArray[i]):
                    maxValue = signalArray[i]
            i+=1
        return signalDemodulated

    def awgnAdd(self,snr_db): 
        self.signalPower = np.array(self.signalModulated)**2
        self.signalPowerDb = 10 * np.log10(self.signalPower)
        sig_avg = abs(np.mean(self.signalPower))
        sig_avg_db = 10 * np.log10(sig_avg)
        noise_avg_db = sig_avg_db - snr_db
        noise_avg = 10 ** (noise_avg_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_avg), len(self.signalModulated))
        signalWithNoise = np.array(self.signalModulated) + noise
        return signalWithNoise


if __name__ == "__main__":

    exampleSignal = DigitalSignal(10)
    exampleSignal.randomSignal(6)
    exampleSignal.ookMod(100)

    plt.figure(1)
    plt.plot(exampleSignal.modTime,exampleSignal.signalModulated)
    plt.title("Señal modulada")
    plt.xlabel("f(t)")
    plt.ylabel("tiempo")   

    noiseExample = exampleSignal.awgnAdd(0.5)
    demExample = exampleSignal.demodulate(exampleSignal.modTime,noiseExample)
    #print(firstSignal.signalDemod)

    plt.figure(2)
    plt.plot(exampleSignal.modTime, noiseExample)
    plt.title('Signal with noise')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')

    snr_levels = [1,1.2,2,2.5,3,4,5,7,9,10]
    bps = [5,1,10]
    for b in bps:
        if b == bps[0]:
            firstSignal = DigitalSignal(b)
            firstSignal.randomSignal(10**4)
        else:
            firstSignal.setBps(b)
        
        firstSignal.ookMod(10)

        for level in snr_levels:
            noiseSignal = firstSignal.awgnAdd(level)
            demSignal = firstSignal.demodulate(firstSignal.modTime,noiseSignal)
            ber = calculateBER(firstSignal.signal,demSignal)
            print(ber)

    
    plt.show()



