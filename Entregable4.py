import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar la base de conocimientos
data = pd.read_csv('tiemporutas.txt', sep=',', header=None, names=['Origen', 'Destino', 'Minutos_Viaje'])

# Convertir el origen y el destino en variables categóricas
data['Origen'] = data['Origen'].astype('category')
data['Destino'] = data['Destino'].astype('category')

# Asignar un codigo a cada origen y destino
data['Origen_code'] = data['Origen'].cat.codes
data['Destino_code'] = data['Destino'].cat.codes

# Pasar la información converida en numeros al cluster
X = data[['Origen_code', 'Destino_code', 'Minutos_Viaje']]

# Agrupamos la información en 10 clusters 
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters)

# Ajustar el modelo
kmeans.fit(X)

# Agregar los labels de los clusters al DataFrame
data['Cluster'] = kmeans.labels_

# Mostrar el resultado
print("\nDatos con clusters:")
print(data.head())

# Reducir la dimensionalidad a 2D para visualizar
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Graficar
plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=data['Cluster'], cmap='viridis', marker='o')
plt.title('Clustering de Rutas')
plt.xlabel('Origen')
plt.ylabel('Destino')
plt.colorbar(label='Cluster')
plt.grid()
plt.show()
