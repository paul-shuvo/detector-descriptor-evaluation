# from skimage import data
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, transform
from skimage.color import rgb2gray
tform = transform.EuclideanTransform(
    rotation=np.pi / 12.,
    translation = (100, -20)
    )
original = data.astronaut()
# grayscale = rgb2gray(original)
#
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# ax = axes.ravel()

# ax[0].imshow(original)
# ax[0].set_title("Original")
# ax[1].imshow(grayscale, cmap=plt.cm.gray)
# ax[1].set_title("Grayscale")

# fig.tight_layout()
transformed = transform.EuclideanTransform(translation=(5,6))
plt.imshow(original)
plt.imshow(transformed)
# print(data)
plt.show()