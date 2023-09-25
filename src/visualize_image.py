import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob
import cv2

img_files = sorted(glob.glob("../data/img/*.png"))
idx = 0

image = cv2.imread(img_files[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 1, 1)

ax1.set_title("Image", fontsize=30)
ax1.imshow(image)

fig.savefig("000031_cvt.jpg")

