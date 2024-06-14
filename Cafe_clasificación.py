

# Importaciones para evaluación de modelos y métricas
from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_score, KFold, train_test_split


# Importaciones para modelos de aprendizaje automático
from sklearn.ensemble import RandomForestClassifier

# Importaciones para manipulación de imagenes y visualización de datos
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import cv2  
import math
from scipy import ndimage


def caracteristicas(imo):

    # Filtro Gaussiano
    img = cv2.GaussianBlur(imo, (7, 7), 0)

    # Conversión a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Conversión a diferentes espacios de color
    XYZ = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    # Segmentación usando umbralización OTSU
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Etiquetado de componentes conectados
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    labels = output[1]
    stats = output[2]
    
    # Selecciono la máscara del objeto más grande
    largest = np.argmax(stats[1:, 4]) + 1
    mascara = (largest == labels)
    mascara = ndimage.binary_fill_holes(mascara).astype(int)
    
    # Contornos
    contours, _ = cv2.findContours(np.uint8(mascara * 255), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    
    # Calcular características geométricas
    # Área
    A = np.sum(mascara)
    # Perimetro
    P = cv2.arcLength(cnt, True)

    # Relación de redondez O DE FORMA
    R = (math.pi * 4 * A) / (P**2)

    # Compacidad
    C = (2 * math.sqrt(A * math.pi)) / P

    # Fcator de compacidad
    fac = ((A - P) / (2 + 1)) / (A - (2 * math.sqrt(A + 1)))
    
    # Caracteristicas medias de intensidades
    rojo = np.sum(mascara * img[:, :, 0] / 255) / np.sum(mascara)
    verde = np.sum(mascara * img[:, :, 1] / 255) / np.sum(mascara)
    azul = np.sum(mascara * img[:, :, 2] / 255) / np.sum(mascara)
    H = np.sum(mascara * hsv[:, :, 0] / 179) / np.sum(mascara)
    S = np.sum(mascara * hsv[:, :, 1] / 255) / np.sum(mascara)
    L = np.sum(mascara * hsv[:, :, 2]) / np.sum(mascara)
    LA = np.sum(mascara * XYZ[:, :, 0]) / np.sum(mascara)
    Ajj = np.sum(mascara * XYZ[:, :, 1]) / np.sum(mascara)
    B = np.sum(mascara * XYZ[:, :, 2]) / np.sum(mascara)

    # Caracteristicas desviaciones estandar
    des0 = np.std(mascara * hsv[:, :, 0])
    des1 = np.std(mascara * hsv[:, :, 1])
    des2 = np.std(mascara * hsv[:, :, 2])
    
    # Caracteristicas a partir de histogramas
    def calcular_hist(imagen, canal):
        hist = cv2.calcHist([imagen], [canal], None, [100], [0, 256])
        return list(itertools.chain(*hist.tolist()))
    
    # Calcular histogramas

    histograma_HSV = calcular_hist(hsv, 0)
    histograma_RGB = calcular_hist(img, 1)
    histograma_XYZ = calcular_hist(XYZ, 0)
    histograma_YCbCr = calcular_hist(ycbcr, 0)
    histograma_HSV1 = calcular_hist(hsv, 1)
    histograma_Ycbcr1 = calcular_hist(ycbcr, 1)
    histograma_YCbCr2 = calcular_hist(ycbcr, 2)
    histograma_RGB2 = calcular_hist(img, 2)
    
    # Concatenar todas las características en una lista final
    datosfinales = [rojo, verde, azul, R, H, S, L, LA, Ajj, B, cnt, R, C, fac, des0, des1, des2]
    listafinal=datosfinales+histograma_HSV + histograma_RGB+histograma_XYZ +histograma_YCbCr +histograma_HSV1+histograma_Ycbcr1 +histograma_YCbCr2 +histograma_RGB2 
    
    return listafinal


datos=[]
etiquetas=[]


for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("f"+str(i)+".jpg")))
    etiquetas.append(1)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("ne"+str(i)+".jpg")))
    etiquetas.append(2)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("in"+str(i)+".jpg")))
    etiquetas.append(3)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("v"+str(i)+".jpg")))
    etiquetas.append(4)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("ce"+str(i)+".jpg")))
    etiquetas.append(5)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("hon"+str(i)+".jpg")))
    etiquetas.append(6)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("mor"+str(i)+".jpg")))
    etiquetas.append(7)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("a"+str(i)+".jpg")))
    etiquetas.append(8)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("s"+str(i)+".jpg")))
    etiquetas.append(9)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("boba"+str(i)+".jpg")))
    etiquetas.append(10)
for i in range(1,101):
    datos.append(caracteristicas(cv2.imread("n"+str(i)+".jpg")))
    etiquetas.append(11)     

# Filtrado de caracteristicas por medio de eliminación recursiva de caracteristicas
hypo=np.array([True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,False,False,False,False,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,True,False,True,True,True,True,True,True,True,True,True,True,False,False,False,False,True,True,True,True,True,True,True,False,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,False,True,False,False,True,False,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,False,True,True,True,True,True,True,True,True,True,True,False,True,True,False,True,True,True,True,False,True,True,True,True,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True,False,True,False,True,False,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])
datos=np.array(datos)

etiquetas=np.array(etiquetas)
datos = pd.DataFrame(datos)

for j in range(817):
    if hypo[j]==False:
        datos.drop([j], axis=1, inplace=True)

        
# Separando los grupos de datos de entrenamiento y validacion
X_train,X_test,y_train,y_test=train_test_split(datos,etiquetas,test_size=0.2, random_state=np.random)

# Implementación del modelo
modelo = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
modelo.fit(X_train,y_train)
predictions = modelo.predict(X_test)

# Validación cruzada
cv = KFold(n_splits=5, random_state=1, shuffle=True)
scores = cross_validate(modelo, datos, etiquetas, cv=cv, scoring={'f1_score': make_scorer(f1_score, average='macro'),'precision_score': make_scorer(precision_score, average='macro'),'recall_score': make_scorer(recall_score, average='macro'),'accuracy_score': make_scorer(accuracy_score)})
  
print(f"Test Set Accuracy : {accuracy_score( y_test, predictions) * 100} %\n\n") 
print(f"Classification Report : \n\n{classification_report( y_test, predictions)}") 

# Matriz de confusión

def graficaconfusion(y_test, predictions):
    matrix=confusion_matrix(y_test,predictions)

    dataframe = pd.DataFrame(matrix,index=['flotador','negro','inmaduro','vinagre','cereza','hongos','mordido','arrugado','sintrilla','broca','normal'],columns=['flotador','negro','inmaduro','vinagre','cereza','hongos','mordido','arrugado','sintrilla','broca','normal'])

    plt.figure(2)
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    h=plt.show()
    return h
h=graficaconfusion(y_test, predictions)