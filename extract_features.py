import numpy as np
import cv2
import os
from skimage.feature import greycomatrix,greycoprops
import pandas as pd
INPUT_SCAN_FOLDER='E:\\Udacity\\steelplate-data\\NEU surface defect database\\' # path where the dataset is stored

slices=[]
for dirName, subdirList, fileList in os.walk(INPUT_SCAN_FOLDER):
        for filename in fileList:
            if ".bmp" in filename.lower():
                slices.append(cv2.imread(os.path.join(dirName, filename),0))

print("done")
proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
featlist= ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','Label']
properties =np.zeros(5)
glcmMatrix = []
final=[]
for i in range(len(slices)):
    img = slices[i]

    # pyplot.imshow((images[k,:,:]),cmap='gray')
    # pyplot.show()
    #  get properties
    glcmMatrix=(greycomatrix(img, [1], [0], levels=256))

   # print(len(glcmMatrix))
    # get properties
    for j in range(0, len(proList)):
        properties[j]=(greycoprops(glcmMatrix, prop=proList[j]))

    label=(i/300)
    features = np.array([properties[0],properties[1],properties[2],properties[3],properties[4],np.floor(label)])
    final.append(features)

df = pd.DataFrame(final,columns=featlist)
filepath="E:\\Udacity\\steelplate-data\\"+"features.xlsx"  #path where to save the features
df.to_excel(filepath)








