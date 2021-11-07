import PIL.Image as img
import numpy as np

a = img.open('./val/274_HC.png').convert('RGB')
b = np.array(a)
print(b)