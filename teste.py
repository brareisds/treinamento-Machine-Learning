from skimage import feature
import numpy as np
import cv2
import os
import csv

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns

        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
       
    
        hist_lbp = [0] * 256
        for linha in lbp:
            print(linha)
            for coluna in linha:
                hist_lbp[int(coluna)] += 1

        print(f"histograma lbp: {hist_lbp}")
        
        # (hist, _) = np.histogram(lbp.ravel(),
		# 	bins=np.arange(0, self.numPoints + 3),
		# 	range=(0, self.numPoints + 2))
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten()  # Converte o histograma em um array unidimensional


        # print(lbp.ravel())  # Print the flattened LBP matrix
        # print(lbp)
        # num_elements = len(lbp.ravel())
        # print("Número de elementos na matriz LBP:", num_elements)
        # max_element = np.max(lbp.ravel())
        # print("Maior elemento na matriz LBP:", max_element)

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)


        # return the histogram of Local Binary Patterns
        return hist


imagePath = "PKLot/PKLotSegmented/PUC/Cloudy/2012-09-12/Occupied/2012-09-12_06_20_57#072.jpg"


desc = LocalBinaryPatterns(8, 1)
# print(desc)
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])
# hist, _ = np.histogram(patterns, bins=np.arange(2**8 + 1), density=True)
hist = desc.describe(gray)
import matplotlib.pyplot as plt

plt.bar(range(len(hist)), hist)
print(range(len(hist)))
plt.show()

