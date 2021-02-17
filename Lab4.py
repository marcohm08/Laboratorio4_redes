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
    def ookMod(self,fc):
        t = [] # arreglo de tiempo de la señal
        init = 0 # tiempo de inicio de la señal
        i = 0
        self.sampleStep = self.bitTime/(10*fc)
        for bit in self.signal: # por cada bit de la señal digital
            if(self.time[i] == self.time[-1]):
                tn = np.arange(init,self.time[i] + 2*self.sampleStep, self.sampleStep)
            else:     
                tn = np.arange(init,self.time[i], self.sampleStep)
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
    def demodulate(self):
        interpolatedF = interpolate.interp1d(self.modTime, self.signalModulated)
        maxValue = 0
        i = 0
        for step in self.modTime:
            if step in self.time:
                print(maxValue)
                if (maxValue >= A):
                    self.signalDemod.append(1)
                else: 
                    self.signalDemod.append(0)
                maxValue = abs(self.signalModulated[i])
            else:
                if(maxValue < self.signalModulated[i]):
                    maxValue = self.signalModulated[i]
            i+=1


        ''' interp = interpolatedF(timeArr)
        for result in interp:
            #print(result)
            if (result == 0):
                self.signalDemod.append(0)
            else:
                self.signalDemod.append(1) '''
    def awgnAdd(self,SNR):
        snr_db = 10 * np.log10(SNR)
        self.signalPower = np.array(self.signalModulated)**2
        self.signalPowerDb = 10 * np.log10(self.signalPower)
        sig_avg = abs(np.mean(self.signalPower))
        sig_avg_db = 10 * np.log10(sig_avg)
        noise_avg_db = sig_avg_db - snr_db
        noise_avg = 10 ** (noise_avg_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_avg), len(self.signalModulated))
        self.signalModulated = np.array(self.signalModulated) + noise
        print(0)
        #print(self.noiseSignal)



if __name__ == "__main__":

    firstSignal = DigitalSignal(10)
    firstSignal.randomSignal(6)
    firstSignal.ookMod(100)
    
    #print(firstSignal.bitTime)
    #print(firstSignal.signal)
    #print(firstSignal.time)

    plt.figure(1)
    plt.plot(firstSignal.modTime,firstSignal.signalModulated)
    plt.title("Señal modulada")
    plt.xlabel("f(t)")
    plt.ylabel("tiempo")   

    firstSignal.awgnAdd(1)
    firstSignal.demodulate()
    print(firstSignal.signalDemod)

    plt.figure(2)
    plt.plot(firstSignal.modTime, firstSignal.signalModulated)
    plt.title('Signal with noise')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')


    plt.show()



