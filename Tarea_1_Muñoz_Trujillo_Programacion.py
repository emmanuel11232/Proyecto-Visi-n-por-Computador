#Se importan las librerías necesarias

from skimage import io, exposure, color
import skimage
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
import scipy.ndimage as ndi

#Se cargan las imágenes tomadas
img_AltoContraste= io.imread(r"C:\Users\emman\Documents\TEC\Sistemas de Visión\Primera Tarea\Alto Contraste.jpg")
img_BCBrilloAlto= io.imread(r"C:\Users\emman\Documents\TEC\Sistemas de Visión\Primera Tarea\Bajo Contraste Brillo Alto.jpg")
img_BCBrilloBajo= io.imread(r"C:\Users\emman\Documents\TEC\Sistemas de Visión\Primera Tarea\Bajo contraste.jpg")
img_Saturado= io.imread(r"C:\Users\emman\Documents\TEC\Sistemas de Visión\Primera Tarea\Saturada 1.jpg")
img_MuySaturada= io.imread(r"C:\Users\emman\Documents\TEC\Sistemas de Visión\Primera Tarea\Muy Saturada.jpg")

#Se crea una función para mostrar varias imágenes al mismo tiempo
def mostrar_imagenes(imagenes, titulos,cmap2=None):
    for j in range(0,2):
      for i in range(len(imagenes)):
          plt.subplot(math.ceil(len(imagenes)/3), 3, i + 1)
          plt.imshow(imagenes[i],cmap=cmap2)
          plt.title(titulos[i])
          plt.axis('off')
    plt.show()

#Se crea una función para mostrar el histograma de una imagen
def mostrar_histograma(imagen, titulo):
  histogram, bin_edges = np.histogram(imagen, bins=256)
  plt.plot(bin_edges[0:-1], histogram)
  plt.xlim(0,1)
  plt.title(titulo)
  plt.xlabel('Valor en escala de grises')
  plt.ylabel('Cantidad de Pixeles')
  plt.show()

#Se muestran las imágenes iniciales
mostrar_imagenes([img_AltoContraste,img_BCBrilloAlto,img_BCBrilloBajo,img_Saturado,img_MuySaturada],["Imagen Alto Contraste","Imagen Brillo Alto", "Imagen Brillo Bajo","Imagen Saturado","Imagen Muy Saturada"])

#Se pasa cada una de las imágenes a escala de grises
gray_AltoCont = skimage.color.rgb2gray(img_AltoContraste)
gray_BrilloAlto = skimage.color.rgb2gray(img_BCBrilloAlto)
gray_BrilloBajo = skimage.color.rgb2gray(img_BCBrilloBajo)
gray_Saturado = skimage.color.rgb2gray(img_Saturado)
gray_MuySaturada = skimage.color.rgb2gray(img_MuySaturada)

#Mostrar histograma para la imagen en escala de grises
mostrar_imagenes([gray_AltoCont,gray_BrilloAlto, gray_BrilloBajo,gray_Saturado,gray_MuySaturada],["Grises Alto Contraste","Grises Brillo Alto", "Grises Brillo Bajo","Grises Saturado","Grises Muy Saturada"],cmap2="gray")
mostrar_histograma(gray_AltoCont,"Histograma Alto Contraste")
mostrar_histograma(gray_BrilloAlto,"Histograma Bajo Contraste Brillo Alto")
mostrar_histograma(gray_BrilloBajo,"Histograma Bajo Contraste Brillo Bajo")
mostrar_histograma(gray_Saturado,"Histograma Saturada")
mostrar_histograma(gray_MuySaturada,"Histograma Muy Saturada")

#Primera imagen
#Para esta primera imagen, se denota alto contraste con valores significativos a lo largo de todo el histograma
#Se denota que debido al brillo y al contraste que se maneja, se tiene que no se debe realizar ningún cambio

#Segunda imagen
#Para esta se denota un bajo contraste y brillo, primeramente, 
#se realizará la técnica de stretching para lograr aumentar el contraste de la imagen


#Se ajustan parámetros del ploteo
matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256,xlabel="Intensidad de pixel",cmap2="gray"):
    #Esta función está basada en un ejemplo mostrado en Scikit-image.Org
    image = skimage.img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()
   

    # Display image
    ax_img.imshow(image, cmap=cmap2)
    #ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    return ax_img, ax_hist, ax_cdf


p2, p98 = np.percentile(gray_BrilloAlto, (2, 98))
#Se realiza la mejora de la imagen mediante stretching
img_Stretching_BCAB = exposure.rescale_intensity(gray_BrilloAlto, in_range=(p2, p98))

#Se realiza la mejora de la imagen mediante ecualización adaptativa
img_adapteq_BCAB = exposure.equalize_adapthist(gray_BrilloAlto, clip_limit=0.03)

#Se realiza la mejora de la imagen mediante ecualización
img_eq_BCAB = exposure.equalize_hist(gray_BrilloAlto)

#Se muestran los resultados 
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=object)

axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5 + i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(gray_BrilloAlto, axes[:, 0])
ax_img.set_title('Bajo Contraste Brillo Alto')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Número de pixeles')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_Stretching_BCAB, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq_BCAB, axes[:, 2])
ax_img.set_title('Ecualización de histograma')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq_BCAB, axes[:, 3])
ax_img.set_title('Ecualización Adaptativa')

ax_cdf.set_ylabel('Fracción de intensidad total')
ax_cdf.set_yticks(np.linspace(0, 1, 5))
fig.tight_layout()
plt.show()

#Para la tercera imagen se presenta también que el problema principal para tener mayores detalles es el contraste, por lo que se genera una
#imagen nueva mediante un stretching de histograma
#Se realiza la mejora mediante el stretching del histograma
p2, p98 = np.percentile(gray_BrilloBajo, (2, 98))
img_Stretching_BCBB = exposure.rescale_intensity(gray_BrilloBajo, in_range=(p2, p98))

#Se realiza la mejora mediante ecualización adaptativa
img_adapteq_BCBB = exposure.equalize_adapthist(gray_BrilloBajo, clip_limit=0.03)

#Se realiza la mejora mediante ecualización
img_eq_BCBB = exposure.equalize_hist(gray_BrilloBajo)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=object)

axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5 + i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(gray_BrilloBajo, axes[:, 0])
ax_img.set_title('Bajo Contraste Brillo Bajo')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Número de pixeles')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_Stretching_BCBB, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq_BCBB, axes[:, 2])
ax_img.set_title('Ecualización de histograma')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq_BCBB, axes[:, 3])
ax_img.set_title('Ecualización Adaptativa')

ax_cdf.set_ylabel('Fracción de intensidad total')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

fig.tight_layout()
plt.show()

#Para las últimas dos imágenes se tiene que ambas tienen saturación, esto se puede corregir mediante una ecualización adaptativa

#Se realiza la mejora mediante el stretching del histograma
img_Stretching_Sat = exposure.rescale_intensity(gray_Saturado, in_range=(p2, p98))

#Se realiza la mejora mediante ecualización adaptativa
img_eq_Saturada = exposure.equalize_hist(gray_Saturado)

#Se realiza la mejora mediante ecualización
img_adapteq_Saturada = exposure.equalize_adapthist(gray_Saturado, clip_limit=0.03)


# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=object)

axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5 + i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(gray_Saturado, axes[:, 0])
ax_img.set_title('Imagen Saturación Moderada')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Número de pixeles')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_Stretching_Sat, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq_Saturada, axes[:, 2])
ax_img.set_title('Ecualización de histograma')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq_Saturada, axes[:, 3])
ax_img.set_title('Ecualización Adaptativa')

ax_cdf.set_ylabel('Fracción de intensidad total')
ax_cdf.set_yticks(np.linspace(0, 1, 5))
fig.tight_layout()
plt.show()

#Se realiza lo mismo para la muy Saturada
#Se realiza la mejora mediante stretching del histograma
img_Stretching_Sat = exposure.rescale_intensity(gray_MuySaturada, in_range=(p2, p98))

#Se realiza la mejora mediante ecualización
img_eq_Muy_Sat= exposure.equalize_hist(gray_MuySaturada)

#Se realiza la mejora mediante ecualización adaptativa
img_adapteq_Muy_Sat = exposure.equalize_adapthist(gray_MuySaturada, clip_limit=0.03)


# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=object)

axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5 + i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(gray_MuySaturada, axes[:, 0])
ax_img.set_title('Saturación Alta')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Número de pixeles')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_Stretching_Sat, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq_Muy_Sat, axes[:, 2])
ax_img.set_title('Ecualización de histograma')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq_Muy_Sat, axes[:, 3])
ax_img.set_title('Ecualización Adaptativa')

ax_cdf.set_ylabel('Fracción de intensidad total')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

fig.tight_layout()
plt.show()

###########################################Segundo objetivo########################################################

#Se pasan las imágenes al modelo de color HSI
img_AltoContHSI=skimage.color.rgb2hsv(img_AltoContraste)
img_BajoContHSI=skimage.color.rgb2hsv(img_BCBrilloBajo)
img_BajoContABHSI=skimage.color.rgb2hsv(img_BCBrilloAlto)
img_SaturadaHSI=skimage.color.rgb2hsv(img_Saturado)
img_MuySaturadaHSI=skimage.color.rgb2hsv(img_MuySaturada)

#Se aisla el canal de tono
Hue_Img1 = img_AltoContHSI[:, :, 0]
Hue_Img2= img_BajoContHSI[:, :, 0]
Hue_Img3 = img_BajoContABHSI[:, :, 0]
Hue_Img4 = img_SaturadaHSI[:, :, 0]
Hue_Img5 = img_MuySaturadaHSI[:, :, 0]

#Se crea una función para obtener el histograma de una sección concreta de la imagen y 
#dibujar un rectángulo negro en esta misma sección
def get_section_histogram(image, section):

    image_Aux=image
    (x1, y1), (x2, y2) = section

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_Aux.shape[1], x2)
    y2 = min(image_Aux.shape[0], y2)

    # Extraer la sección de la imagen
    section_image = image_Aux[y1:y2, x1:x2]

    # Crear histograma de la sección de la imagen
    histogram, bin_edges = np.histogram(section_image, bins=256)

    #Extraer coordenadas de la sección como tupla
    (x1, y1), (x2, y2) = section

    #Dibujar el rectángulo
    rr, cc = skimage.draw.rectangle_perimeter(start=(y1, x1), end=(y2, x2), shape=image.shape, clip=True)
    image_Aux[rr, cc] = 0
    #Se devuelve el histograma de la sección, la imagen con el cuadrado y los rangos de ploteo
    return histogram,bin_edges,image_Aux


#Se crea una función para realizar la binarización de una imagen, permite realizarse por
#Otsu, meter varios rangos y enviar estos rangos a blanco o a negro
def Binarization(rangos,image,To_Black=False,Otsu=False):
  Image_Aux=image
  mask2=np.zeros_like(Image_Aux)
  mask1=np.zeros_like(Image_Aux)
  if not Otsu:
    for i in rangos:
      mask1 = np.logical_and(Image_Aux < i[1],Image_Aux > i[0])
      mask2 = np.logical_or(mask1,mask2)
  else:
    thresh = skimage.filters.threshold_otsu(image)
    mask2= Image_Aux > thresh
  if To_Black:
      mask2=np.logical_not(mask2)
  return mask2

#Primera imagen

#Se obtiene el histograma de la sección 
Hist_AC_HSI,Bin1,Square_AC=get_section_histogram(Hue_Img1,((0,0),(200,1250)))

#Se plotea este histograma y la sección utilizada
plt.plot(Bin1[0:-1], Hist_AC_HSI)
plt.title('Histograma Sección Alto Contraste Tono')
plt.show()

plt.imshow(Square_AC,cmap="gray")
plt.title('Sección de Histograma (Cuadrado negro)')
plt.show()

#Se crea la máscara binarizada a partir del rango visto
Seg_Mask_AC=Binarization([[0.4,0.55]],Hue_Img1,To_Black=True)

#Se plotea la máscara para poder ver el resultado de la binarización
plt.imshow(Seg_Mask_AC, cmap='gray')
plt.title('Máscara a partir de seción')
plt.show()

#Se suaviza la máscara anterior mendiante umbralización adaptativa
block_size = 31
local_thresh_AC = skimage.filters.threshold_local(Seg_Mask_AC, block_size,method="median")

#Se plotea la máscara final ya suavizada
plt.imshow(local_thresh_AC, cmap='gray')
plt.title('Máscara suavizada')
plt.show()

#Segunda imagen
#Se obtiene el histograma de la sección 
Hist_BCBB_HSI,Bin2,Square_BCBB=get_section_histogram(Hue_Img2,((0,0),(200,1450)))

#Se plotea este histograma y la sección utilizada
plt.imshow(Square_BCBB,cmap="gray")
plt.title('Sección para Histograma (Cuadrado negro)')
plt.show()

plt.plot(Bin2[0:-1], Hist_BCBB_HSI)
plt.title('Histograma de Tono de Sección, Bajo Brillo ')
plt.show()

#Se crea la máscara binarizada a partir del rango visto
Seg_Mask_BCBB=Binarization([[0.3,0.53]],Hue_Img2,To_Black=True)

#Se plotea la máscara para poder ver el resultado de la binarización
plt.imshow(Seg_Mask_BCBB, cmap='gray')
plt.title('Mascara a partir de seccion')
plt.show()

#Se suaviza la máscara anterior mendiante umbralización adaptativa
block_size=25
local_thresh_BCBB = skimage.filters.threshold_local(Seg_Mask_BCBB, block_size,method="median")
plt.imshow(local_thresh_BCBB, cmap='gray')
plt.title('Máscara Suavizada Bajo Brillo')
plt.show()

#Tercera imagen

#Se obtiene el histograma de la sección 
Hist_BCAB_HSI,Bin3,Square_BCAB=get_section_histogram(Hue_Img3,((750,0),(900,1200)))

#Se plotea este histograma y la sección utilizada
plt.imshow(Square_BCAB,cmap="gray")
plt.title('Sección para Histograma (Cuadrado negro)')
plt.show()

plt.plot(Bin3[0:-1], Hist_BCAB_HSI)
plt.title('Histograma de Tono de Sección, Brillo Alto')
plt.show()


#Se crea la máscara binarizada a partir del rango visto
Seg_Mask_BCBB=Binarization([[0.44,0.5]],Hue_Img3,To_Black=True)

#Se plotea la máscara para poder ver el resultado de la binarización
plt.imshow(Seg_Mask_BCBB, cmap='gray')
plt.title('Máscara de a partir de sección')
plt.show()

#Se suaviza la máscara anterior mendiante umbralización adaptativa
block_size=51
local_thresh_BCAB = skimage.filters.threshold_local(Seg_Mask_BCBB, block_size,method="median")
plt.imshow(local_thresh_BCAB, cmap='gray')
plt.title('Máscara Suavizada')
plt.show()


#Cuarta imagen

#Se obtiene el histograma de la sección 
Hist_MSat_HSI,Bin4,Square_MSat=get_section_histogram(Hue_Img4,((0,1380),(1000,1600)))

#Se plotea este histograma y la sección utilizada
plt.imshow(Square_MSat,cmap="gray")
plt.title('Sección para Histograma (Cuadrado negro)')
plt.show()

plt.plot(Bin4[0:-1], Hist_MSat_HSI)
plt.title('Histograma de Tono de Sección, Saturación Moderada')
plt.show()

#Se crea la máscara binarizada a partir del rango visto
Seg_Mask_MSat=Binarization([[0.3,0.5]],Hue_Img4,To_Black=True)
#Se plotea la máscara para poder ver el resultado de la binarización
plt.imshow(Seg_Mask_MSat, cmap='gray')
plt.title('Máscara a partir de sección')
plt.show()

#Se suaviza la máscara anterior mendiante umbralización adaptativa
block_size=41
local_thresh_MSat = skimage.filters.threshold_local(Seg_Mask_MSat, block_size,method="median")
plt.imshow(local_thresh_MSat, cmap='gray')
plt.title('Máscara Suavizada')
plt.show()

#Quinta imagen

#Se obtiene el histograma de la sección 
Hist_Sat_Alta_HSI,Bin5,Square_Sat_Alta=get_section_histogram(Hue_Img5,((0,0),(200,1280)))

#Se plotea este histograma y la sección utilizada
plt.plot(Bin5[0:-1], Hist_Sat_Alta_HSI)
plt.title('Histograma de Tono de Sección, Saturación Alta')
plt.show()

plt.imshow(Square_Sat_Alta,cmap="gray")
plt.title('Sección para Histograma (Cuadrado negro)')
plt.show()

#Se crea la máscara binarizada a partir del rango visto
Seg_Mask_Sat_Alta=Binarization([[0.2,0.55]],Hue_Img5,To_Black=True)
plt.imshow(Seg_Mask_Sat_Alta,cmap="gray")
plt.title('Máscara a partir de sección')
plt.show()

#Se suaviza la máscara anterior mendiante umbralización adaptativa
block_size=41
local_thresh_Sat_Alta = skimage.filters.threshold_local(Seg_Mask_Sat_Alta, block_size,method="median")
plt.imshow(local_thresh_Sat_Alta, cmap='gray')
plt.title('Máscara Suavizada')
plt.show()

#Se generan los resultados

#Se crea una lista de títulos para los histogramas a plotear 
Titulos=["Histograma de intesidad de imagen Bajo Contraste y Bajo Brillo corregida","Histograma de intesidad de imagen con Bajo Contraste y Alto Brillo corregida","Histograma de intesidad de imagen con Saturación Moderada corregida","Histograma de intesidad de imagen con Saturación Alta corregida"]
#Se crea una función que realiza el cambio de la matriz de intensidad en HSI por la matriz de escala de grises
#Luego plotea el histograma de la escala de grises corregida
#Luego recrea la imagen en RGB y la asigna a una lista
def New_Image(HSI: list,Contrast: list):
    RGB_Imgs=[]
    for a in range(0,len(HSI)):
        HSI_Img=HSI[a]
        HSI_Img[:, :, 2]=Contrast[a]
        histogram, bin_edges = np.histogram(HSI_Img[:, :, 2], bins=256)
        plt.plot(bin_edges[0:-1],histogram)
        plt.title(Titulos[a])
        plt.xlabel("Intensidad de pixel")
        plt.ylabel("Número de pixeles")
        plt.show()
        RGB_Img=skimage.color.hsv2rgb(HSI_Img)
        RGB_Imgs.append(RGB_Img)
    return RGB_Imgs

#Se ejecuta la función anterior
RGBs=New_Image([img_BajoContHSI,img_BajoContABHSI,img_SaturadaHSI,img_MuySaturadaHSI],[img_Stretching_BCBB,img_Stretching_BCAB,img_adapteq_Saturada,img_adapteq_Muy_Sat])
#Se adjunta la imagen en alto contraste
RGBs.insert(0,img_AltoContraste)
#Se crea una lista de las máscaras realizadas para hacer la categorización
Masks=[local_thresh_AC,local_thresh_BCBB,local_thresh_BCAB,local_thresh_MSat,local_thresh_Sat_Alta]
Titulos=["Resultados Alto Contraste","Resultados Bajo Contraste Bajo Brillo","Resultados Bajo Contraste Alto Brillo","Resultados Saturación Moderada","Resultados Saturación Alta"]

#Se realiza la categorización de cada una de las imágenes ya corregidas y se plotean para observar los resultados.
for a in range(0,len(Masks)):
    label_image = skimage.measure.label(Masks[a])
    image_label_overlay = skimage.color.label2rgb(label_image, image=RGBs[a], bg_label=0)
    plt.imshow(image_label_overlay)
    plt.title(Titulos[a])
    plt.show()
