from PIL import Image
import numpy as np
a=Image.open(r'D:\SR-GAN\data\600-confocal-new\600c-3-20-7-r9.png')
a=np.array(a)
print(a.shape)