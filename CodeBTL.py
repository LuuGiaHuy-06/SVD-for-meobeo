from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [16, 8]

def plot_images(name ,U, S, VT):
    j = 0
    for r in (5, 25, 100):
        Xapprox = U[:,:r] @ S[0:r, :r] @ VT[:r, :]
        plt.figure(j+1)
        j += 1
        img = plt.imshow(256-Xapprox)
        img.set_cmap('gray')
        plt.axis('off')
        plt.title('r = ' + str(r))
        plt.savefig(f'{name}_{j}.png')
        plt.show()
#not necessary
def plot_cumulative_sum(S):
    plt.clf
    plt.figure(1)
    plt.semilogy(np.diag(S))
    plt.title('Singular Values')
    plt.show()

    plt.figure(1)
    plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
    plt.title('Singular Values: Cumulative Sum')
    plt.show()


A = imread('Meobeo.png')
X = np.mean(A, -1); 
U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

plot_images("Meobeo" ,U, S, VT)
plot_cumulative_sum(S)
