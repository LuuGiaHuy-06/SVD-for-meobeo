import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from PIL import Image
"""
Singular Value Decomposition
"""
def find_eigenvector(parameter):
    eigenvalues, eigenvectors = np.linalg.eig(parameter)
    return eigenvectors

def find_sigma(A):
    AT = np.transpose(A)
    AAT = np.dot(A,AT) 
    ATA = np.dot(AT,A) 
    Q = find_eigenvector(AAT)
    P = find_eigenvector(ATA)
    sigma = np.dot(np.dot(np.linalg.inv(Q),A),P)
    sigma = np.round(sigma, 2)
    return sigma

#A = np.array([[3, 2, 2], [2, 3, -2]])
#print(find_sigma(A))
#U, S, Vh = np.linalg.svd(find_sigma(A), full_matrices=True)

 
 
img = Image.open('Meobeo.png')
print(img.mode, img.size)
numpydata = np.asarray(img)
numpydata = np.round(numpydata,0)

def iterative(matrix):
    U, sigma, VT = np.linalg.svd(matrix)
    threshold = 50
    keep_indices = (sigma >= threshold)
    sigma_filtered = sigma[keep_indices]
    A_reduced = np.dot( (U, np.diag(sigma_filtered)) , np.transpose(VT) )  
    A_reduced, sigma_filtered
    return A_reduced

pilImage = Image.fromarray(iterative(numpydata))
print(type(pilImage))
print(pilImage.mode, pilImage.size)
imga = img.save("geeks.png")

"""
for i in [1,2,3]:
    pilImage = Image.fromarray(iterative(pilImage))
    print(type(pilImage))
    print(pilImage.mode, pilImage.size)
    imga = img.save("geeks.png")
"""
