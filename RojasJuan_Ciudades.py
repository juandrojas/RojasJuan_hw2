#!/usr/bin/env python3
#importamos paquetes
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image #permite carga de archivos 'jpg'

#'import_image_grayscale' recibe un argumento 'path' que es la ruta de una imagen a importar en escala de grises
def import_image_grayscale(path):
    return Image.open(path).convert('L') #convert('L') convierte la imagen a escala de grises

#se importan las imagenes
paris = import_image_grayscale('Paris.jpg')
barc = import_image_grayscale('Barcelona.jpg')
triangulos = import_image_grayscale('triangulos.png')
frac = import_image_grayscale('frac.jpeg')

#'fourier_2D' retorna la transformada de Fourier de la imagen 'img' y un shifted array de las magniitudes del anterior
def fourier_2D(img):
    F = np.fft.fft2(img)
    return np.fft.fftshift(F), abs(np.fft.fftshift(F)) #fftshift hace un 'shifting' a la transformada de Fourier, para la gráfica sea mucho más clara

#calculamos Fourier_2D para nuestras imagenes
paris_fft2, paris_fft2_amplitude = fourier_2D(paris)
barc_fft2, barc_fft2_amplitude = fourier_2D(barc)
triangulos_fft2, triangulos_fft2_amplitude = fourier_2D(triangulos)
frac_fft2, frac_fft2_amplitude = fourier_2D(frac)

#graficamos nuestras imagenes
fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2,2, figsize=(15,15))
fig.suptitle('Imagenes', fontsize=18)
ax11.imshow(paris)
ax11.set_title('Paris')
ax12.imshow(barc)
ax12.set_title('Barcelona')
ax21.imshow(triangulos)
ax21.set_title('Triangulos')
ax22.imshow(frac, cmap='gray')
ax22.set_title('Fractales')
plt.subplots_adjust(hspace=0.5)
fig.savefig('imagenes.pdf')
plt.gcf().clear()
plt.close()

#graficamos sus transformadas de Fourier
fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2,2, figsize=(15,15))
fig.suptitle('Transformadas de Fourier', fontsize=24)
ax11.imshow(np.log(paris_fft2_amplitude), cmap='gray')
ax12.imshow(np.log(barc_fft2_amplitude), cmap='gray')
ax21.imshow(np.log(triangulos_fft2_amplitude), cmap='gray')
ax22.imshow(np.log(frac_fft2_amplitude), cmap='gray')
ax22.set_title('Fractales - magnitud del espectro')
ax21.set_title('Triangulos - magnitud del espectro')
ax12.set_title('Barcelona - magnitud del espectro')
ax11.set_title('Paris - magnitud del espectro')
plt.subplots_adjust(hspace=0.5)
fig.savefig('transformadas.pdf')
plt.gcf().clear()
plt.close()

#'hor_cut' realiza un corte transeversal a un arreglo de magnitudes de una transformada de Fourier 'fft2_mag'
def hor_cut(fft2_mag):
    m = np.shape(fft2_mag)[1]
    return np.arange(m), fft2_mag[m//2]

#graficamos cortes transversales de nuestras imagenes
fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2,2, figsize=(15,15))
fig.suptitle('Cortes transversales', fontsize=24)
ax11.bar(hor_cut(paris_fft2_amplitude)[0],hor_cut(paris_fft2_amplitude)[1], width=5.)
ax12.bar(hor_cut(barc_fft2_amplitude)[0],hor_cut(barc_fft2_amplitude)[1], width=5.)
ax21.bar(hor_cut(triangulos_fft2_amplitude)[0],hor_cut(triangulos_fft2_amplitude)[1], width=5.)
ax22.bar(hor_cut(frac_fft2_amplitude)[0],hor_cut(frac_fft2_amplitude)[1], width=5.)
ax22.set_title('Fractales - Corte transversal')
ax21.set_title('Triangulos - Corte transversal')
ax12.set_title('Barcelona - Corte transversal')
ax11.set_title('Paris - Corte transversal')
plt.subplots_adjust(hspace=0.5)
fig.savefig('cortes_transversales.pdf')
plt.gcf().clear()
plt.close()

# 'rm_hor_road' remueve las calles horizontales de un mapa 'img', con transformada de fourier 'fft2_mag', respetando el cuadro 'w' * 'h' en la gráfica de la transfromada de Fourier
def rm_hor_road(img, fft2_mag, w, h):
    m = np.shape(fft2_mag)[1]
    n = np.shape(fft2_mag)[0]
    output = np.zeros_like(fft2_mag)
    for i in range(n):
        for j in range(m):
            if ( abs((m//2)-j) >= w or abs((n//2)-i) <= h ):
                output[i,j] = fft2_mag[i,j]
    return np.fft.ifft2(output)

#graficamos 'rm_hor_road' para Barcelona.jpg
plt.figure(figsize=(12,8))
plt.imshow(abs(rm_hor_road(barc, barc_fft2, 20, 20)), cmap=plt.cm.gray)
plt.savefig('sin_horizontales.pdf')
plt.close()
