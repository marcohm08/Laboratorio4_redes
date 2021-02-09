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
f_0 = 100 * 10**6 # frecuencia cuando el bit es 0 en Hz
f_1 = 200 * 10**6 # frecuencia cuando el bit es 1 en Hz

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
            init+=self.bitTime
    def ookMod(self,fc):
        t = [] # arreglo de tiempo de la señal
        init = 0 # tiempo de inicio de la señal
        i = 0
        for bit in self.signal: # por cada bit de la señal digital
            if (bit == 0): # si el bit es igual a 0
                tn = np.arange(init,self.time[i] + self.time[i]/100, 1/(2*fc))
                for t_i in tn: # por cada elemento de arreglo de tiempo tn
                    value = 0 # se calcula el coseno en el tiempo t__i
                    self.signalModulated.append(value) # se agrega el valor a arreglo de valores modulados
                    t.append(t_i) # se agrga t_i al arreglo de tiempo der salida
            elif (bit == 1): # En caso de que el bit sea 1 es lo mismo quie en el caso anteriorm pero con 
                # la frecuencia cuando el bit es igual a 1
                tn = np.arange(init,self.time[i] + self.time[i]/100, 1/(2*fc))
                for t_i in tn:
                    value = cosFunction(t_i,fc)
                    self.signalModulated.append(value)
                    t.append(t_i)     
            init = init + self.bitTime
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
        interp = interpolatedF(self.time)
        for result in interp:
            if (result == 0):
                self.signalDemod.append(0)
            else:
                self.signalDemod.append(1)



if __name__ == "__main__":

    firstSignal = DigitalSignal(10)
    firstSignal.randomSignal(10)
    firstSignal.ookMod(100)
    firstSignal.demodulate()
    print(firstSignal.signal)
    print(firstSignal.time)

    plt.figure(1)
    plt.plot(firstSignal.modTime,firstSignal.signalModulated)
    plt.title("Señal modulada")
    plt.xlabel("f(t)")
    plt.ylabel("tiempo")   

    print(firstSignal.signalDemod)

    plt.show()



