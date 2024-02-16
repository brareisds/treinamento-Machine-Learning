import cv2 
import numpy as np 
import os
import csv
from matplotlib import pyplot as plt 

	
def get_pixel(img, center, x, y): 
	
	new_value = 0
	
	try: 
		# If local neighbourhood pixel 
		# value is greater than or equal 
		# to center pixel values then 
		# set it to 1 
		if img[x][y] >= center: 
			new_value = 1
			
	except: 
		# Exception is required when 
		# neighbourhood value of a center 
		# pixel value is null i.e. values 
		# present at boundaries. 
		pass
	
	return new_value 

# Function for calculating LBP 
def lbp_calculated_pixel(img, x, y): 

	center = img[x][y] 

	val_ar = [] 
	
	# top_left 
	val_ar.append(get_pixel(img, center, x-1, y-1)) 
	
	# top 
	val_ar.append(get_pixel(img, center, x-1, y)) 
	
	# top_right 
	val_ar.append(get_pixel(img, center, x-1, y + 1)) 
	
	# right 
	val_ar.append(get_pixel(img, center, x, y + 1)) 
	
	# bottom_right 
	val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
	
	# bottom 
	val_ar.append(get_pixel(img, center, x + 1, y)) 
	
	# bottom_left 
	val_ar.append(get_pixel(img, center, x + 1, y-1)) 
	
	# left 
	val_ar.append(get_pixel(img, center, x, y-1)) 
	
	# Now, we need to convert binary 
	# values to decimal 
	power_val = [1, 2, 4, 8, 16, 32, 64, 128] 

	val = 0
	
	for i in range(len(val_ar)): 
		val += val_ar[i] * power_val[i] 
		
	return val 

path = 'PKLot/PKLotSegmented/PUC/Cloudy/2012-09-12/Occupied/2012-09-12_06_20_57#072.jpg'
img_bgr = cv2.imread(path, 1) 

height, width, _ = img_bgr.shape 

# We need to convert RGB image 
# into gray one because gray 
# image has one channel only. 
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 

# Create a numpy array as 
# the same height and width 
# of RGB image 
img_lbp = np.zeros((height, width), np.uint8) 

for i in range(0, height): 
	for j in range(0, width): 
		img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j) 


print("LBP Program is finished") 

eps=1e-7
hist = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
hist = hist.flatten()  # Converte o histograma em um array unidimensional
hist = hist.astype("float")
hist /= (hist.sum() + eps)

# # Criar a figura e os eixos
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# # Mostrar a imagem BGR no primeiro eixo
# ax1.imshow(img_bgr)
# ax1.set_title('BGR image')
# ax1.axis('off')

# # Mostrar a imagem LBP no segundo eixo
# ax2.imshow(img_lbp, cmap='gray')
# ax2.set_title('LBP Image')
# ax2.axis('off')

# # Mostrar o histograma no terceiro eixo
# ax3.bar(range(len(hist)), hist)
# print(range(len(hist)))
# ax3.set_title('Histograma')

# # Remover os eixos do terceiro subplot
# # ax3.axis('off')

# # Ajustar o espaçamento entre os subplots
# plt.tight_layout()

# # Mostrar a figura
# plt.show()

diretorio_imagens = "PKLot/PKLotSegmented"

# Nome do arquivo CSV
arquivo_csv = 'histograma.csv'


# Abre o arquivo CSV em modo de escrita
with open(arquivo_csv, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for root, _, files in os.walk(diretorio_imagens):
        for filename in files:
            classe = os.path.basename(root)

            if classe == "Occupied":
                classe = 1
            elif classe == "Empty":
                classe = 0

            # Concatena o histograma com a classe
            vetor_caracteristicas = np.concatenate((hist, [classe]))
            
            # Escreve no arquivo CSV
            writer.writerow(vetor_caracteristicas)






