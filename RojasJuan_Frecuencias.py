#importamos paquetes a utilizar
import numpy as np
import matplotlib.pyplot as plt
import csv

#obtenemos datos del archivo 'Signal.csv' en 'data_t' y 'data_y'
data_t = np.array([])
data_y = np.array([])
with open('Signal.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data_t = np.append(data_t, float(row[0]))
        data_y = np.append(data_y, float(row[1]))

#graficamos la señal
plt.plot(data_t, data_y)
plt.title('Señal Original')
plt.savefig('Signal.pdf')
plt.close()

#calculamos tamaño de los datos 'n' y espaciamiento temporal 'dt'
n = np.size(data_t)
dt = data_t[1] - data_t[0]

#esta funciòn realiza la transformada de Fourier discreta para un array 'y'
def disc_four_trans(y):
    fourier = np.array([])
    for m in range(n):
        aux_sum = 0
        for k in range(n):
            aux_sum = aux_sum + y[k]*np.exp(-2j*k*np.pi*m/n)
        fourier = np.append(fourier, aux_sum)
    return fourier

#esta funcion recupera las frecuencias dado 'n' y 'dt'
def recov_freq(n, dt):
    if n%2 == 0:
        return np.append(np.arange(n/2), -np.arange(1,n/2+1)[::-1])/(n*dt)
    else:
        return np.append(np.arange((n-1)/2), -np.arange(1,(n-1)/2)[::-1])/(n*dt)

#recuperamos las frecuencias y transformada de Fourier de nuestros datos
freq = recov_freq(n,dt)
fft = disc_four_trans(data_y)
fft_mag = np.absolute(disc_four_trans(data_y))

#calculamos solamente la parte positiva de las frecuencias para obtener las 3 frecuencias principales
pos_freq = freq[freq >= 0]
pos_fft_mag = fft_mag[freq >= 0]
index = pos_fft_mag.argsort()[-3:]

#mensaje a consola
print("las tres frecuencias principales de la señal son:",str(round(pos_freq[index][-1])) + ',', round(pos_freq[index][-2]), "y",str(round(pos_freq[index][-3])) + '.')

#plotting de la transformada de Fourier se guarda en 'TF_Signal.pdf'
plt.figure()
plt.bar(freq, fft_mag/n, width = 30.)
plt.savefig('TF_Signal.pdf')
plt.gcf().clear()
plt.close()

#'f_c' frecuencia de corte filtro_pasabajos
f_c = 1000.0

#esta función realiza un filtro pasabajos dado por la frecuencia de corte. Recibe 'freq' un array de frecuencias y 'fft' un array de la transformada de Fourier de las frecuencias 'freq'
def filtro_pasabajos(freq, fft):
    return fft*np.array([np.abs(freq) <= f_c]).astype(int).flatten()

#tranformada inversa de Fourier
def inv_disc_fourier_transform(y):
    fourier = np.array([])
    for m in range(n):
        aux_sum = 0
        for k in range(n):
            aux_sum = aux_sum + y[k]*np.exp(2j*k*np.pi*m/n)
        fourier = np.append(fourier, (1/n)*aux_sum)
    return fourier

#plotting 'Sin Filtrar' y plotting con 'Filtro pasabajos'. Se guarda en 'SignalFiltro.pdf'
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False)
plt.subplots_adjust(hspace=0.6)
ax0.plot(data_t, data_y)
ax0.set_title('Sin Filtrar')
ax1.plot(data_t, inv_disc_fourier_transform(filtro_pasabajos(freq, fft)))
ax1.set_title('Filtro pasabajos')
plt.savefig('SignalFiltro.pdf')
