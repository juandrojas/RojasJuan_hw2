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
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(15,35))
ax1.plot(dates, azores)
ax1.set_title('Azores', fontsize=14)
ax2.plot(dates, darwin)
ax2.set_title('Darwin', fontsize=14)
ax3.plot(dates, gibraltar)
ax3.set_title('Gibraltar', fontsize=14)
ax4.plot(dates, iceland)
ax4.set_title('Islandia', fontsize=14)
ax5.plot(dates, madras)
ax5.set_title('Madras', fontsize=14)
ax6.plot(dates, nagasaki)
ax6.set_title('Nagasaki', fontsize=14)
ax7.plot(dates, tahiti)
ax7.set_title('Tahiti', fontsize=14)
fig.suptitle('Presiones Atmosféricas', fontsize=18)
fig.subplots_adjust(top=0.95, bottom=0.05)
fig.savefig('Pressure.pdf')
plt.gcf().clear()
plt.close()

#calculamos matriz de covarianza.
N = 7
data_size = np.size(azores)

#'norm' normaliza 'data'
def norm(data):
    return (data-np.mean(data))

#datos normalizados
n_data = np.array([norm(azores), norm(darwin), norm(gibraltar), norm(iceland), norm(madras), norm(nagasaki), norm(tahiti)])

#'cov_mat' calcula la matriz de covarianza de 'norm_data'. Se asume que 'norm_data' está normalizada.
def cov_mat(norm_data):
    cov = np.empty([N,N])
    for i in range(N):
        for j in range(N):
            cov[i,j] = sum(norm_data[i]*norm_data[j])/data_size
    return cov

cov = cov_mat(n_data)
eigen = np.linalg.eig(cov)
print(eigen[0]/sum(eigen[0]))

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
plt.savefig('PCA.pdf')

#PCA Analysis
