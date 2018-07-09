#paquetes a importar
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
np.set_printoptions(linewidth = 100)

#'import_data' recibe 'path' una ruta al archivo a importar. Retorna un array que elimina la primera columna de los datos (año) y lo covierte de un array 2D a uno 1D.
def import_data(path):
    data = np.loadtxt(path)
    return np.delete(data, 0, axis=1).flatten()

#importamos nuestros datos
azores = import_data('azores.txt')
darwin = import_data('darwin.txt')
gibraltar = import_data('gibraltar.txt')
iceland = import_data('iceland.txt')
madras = import_data('madras.txt')
nagasaki = import_data('nagasaki.txt')
tahiti = import_data('tahiti.txt')

#'ini_year' es el año inicial de nuestra base de datos. 'fn_year' el año final.
ini_year = 1900
fn_year = 1999

#'create_date_arr' crea un array de obejtos 'date' desde el primer mes del año 'ini_year' al último mes del año 'fn_year'.
def create_date_arr(ini_year, fn_year):
    dates = np.array([])
    first = dt.date(ini_year,1,1)
    for y in range(ini_year, fn_year+1):
        for m in range(1,13):
            dates = np.append(dates, first.replace(year=y, month=m))
    return dates

dates = create_date_arr(ini_year,fn_year)

#plotting original data
#fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), ax7) = plt.subplots(4, 2, figsize=(15,35))
plt.figure(figsize=(20,12))
plt.suptitle('Presiones Atmosféricas', fontsize=24)
plt.subplot(421)
plt.plot(dates, azores)
plt.title('Azores', fontsize=20)
plt.subplot(422)
plt.plot(dates, darwin)
plt.title('Darwin', fontsize=20)
plt.subplot(423)
plt.plot(dates, gibraltar)
plt.title('Gibraltar', fontsize=20)
plt.subplot(424)
plt.plot(dates, iceland)
plt.title('Islandia', fontsize=20)
plt.subplot(425)
plt.plot(dates, madras)
plt.title('Madras', fontsize=20)
plt.subplot(426)
plt.plot(dates, nagasaki)
plt.title('Nagasaki', fontsize=20)
plt.subplot(427)
plt.plot(dates, tahiti)
plt.title('Tahiti', fontsize=20)
plt.subplots_adjust(hspace=0.6)
plt.savefig('Pressure.pdf')
plt.gcf().clear()
plt.close()

#calculamos matriz de covarianza.
N = 7
data_size = np.size(azores)

#'norm' normaliza 'data'
def norm(data):
    return (data-np.mean(data))/np.std(data)

#datos normalizados
n_data = np.array([norm(azores), norm(darwin), norm(gibraltar), norm(iceland), norm(madras), norm(nagasaki), norm(tahiti)])

#'cov_mat' calcula la matriz de covarianza de 'norm_data'. Se asume que 'norm_data' está normalizada.
def cov_mat(norm_data):
    cov = np.empty([N,N])
    for i in range(N):
        for j in range(N):
            cov[i,j] = sum(norm_data[i]*norm_data[j])/(data_size-1)
    return cov

cov = cov_mat(n_data)
eigen = np.linalg.eig(cov)

#'accpet_rate' qué tanto porcentaje de información de los datos se desea preservar haciendo PCA.
accept_rate = .75

#dada una matriz de covarianza 'cov', 'choose_rel_data' calcula los componentes princpales quye preservan 'accept_rate'. Lanza una excepción a ratios de aceptación absurdos
def choose_rel_data(cov, accept_rate):
    if accept_rate >= 1:
        raise ValueError('El ratio de aceptación no puede ser mayor o igual a 1.')
    eigen = np.linalg.eig(cov)
    order = eigen[0].argsort()[::-1]
    eigen_val = eigen[0][order]
    eigen_vec = eigen[1][:,order]
    rel_index = -1
    while sum(eigen_val[:rel_index])/sum(eigen_val) >= accept_rate:
        rel_index = rel_index - 1
    if rel_index == -1:
        raise ValueError('El ratio de aceptación es demasiado alto como para filtrar datos.')
    return eigen_val[:rel_index + 1], eigen_vec[:, :rel_index+1]

e_val_pca, e_vec_pca = choose_rel_data(cov, accept_rate)

#mensaje
print("Se escogen las",np.size(e_val_pca),"primeras componentes principales de los datos, pues estas representan más del", str(accept_rate*100)+"%","(específicamente el",str(round(sum(e_val_pca)/sum(np.linalg.eig(cov)[0])*100,2)) + "%)", "de información de los datos.")

#PCA and plotting
compressed_data = np.dot(np.transpose(e_vec_pca), n_data)
plt.figure()
plt.scatter(compressed_data[0],compressed_data[1], s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('PCA.pdf')
plt.close()

#PCA Analysis
#primer componente principal
fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2,2, figsize=(20,12))
loc = [0,'Azores', 'Darwin', "Gibraltar", "Islandia", 'Madras', 'Nagasaki', 'Tahiti']
ax11.set_title('Coordenadas del primer componente principal', fontsize=20)
ax11.stem(e_vec_pca[:,0])
ax11.set_xticklabels(loc, rotation = 45)
plt.subplots_adjust(bottom=0.1, hspace=0.4)
#primer componente scores
ax12.set_title('Valores del primer componente principal', fontsize=20)
ax12.scatter(dates, compressed_data[0])
ax12.set_xlim(dt.date(1895,1,1), dt.date(2005,1,1))
ax12.axhline(0, color='black')
#segundo componente principal
ax21.set_title('Coordenadas del segundo componente principal', fontsize=20)
ax21.stem(e_vec_pca[:,1])
ax21.set_xticklabels(loc, rotation = 45)
#segundo componente scores
ax22.set_title('Valores del segundo componente principal', fontsize=20)
ax22.scatter(dates, compressed_data[1])
ax22.set_xlim(dt.date(1895,1,1), dt.date(2005,1,1))
ax22.axhline(0, color='black')
fig.savefig('graficasAdicionales.pdf')
