import cv2 
import numpy as np 
import os, os.path
import csv
import tqdm
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


print("LBP Program is finished") 


DIR = 'PKLot/PKLotSegmented'
arquivo_csv = 'dados_hist.csv'

with open(arquivo_csv, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for root, _, files in os.walk(DIR):
        for filename in files:
            path = os.path.join(root, filename) 
            print(path)
            
            img_bgr = cv2.imread(path, 1) 
            height, width, _ = img_bgr.shape 
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 
            img_lbp = np.zeros((height, width), np.uint8) 

            for i in range(0, height): 
                for j in range(0, width): 
                    img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j) 

            hist = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
            hist = hist.flatten()

            classe = os.path.basename(root)

            if classe == "Occupied":
                classe = 1
            elif classe == "Empty":
                classe = 0

            vetor_caracteristicas = np.concatenate((hist, [classe]))

            writer.writerow(vetor_caracteristicas)
