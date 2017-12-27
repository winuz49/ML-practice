from PIL import Image
import scipy.misc as misc
import numpy as np

lena = Image.open('input1.png')
# lena.show()

lena_modified = Image.open('input2.png')
# lena_modified.show()

width,heigth = lena.size
ans = Image.new(mode='RGBA', size=lena.size)

lena_color = lena.load()
lena_modified_color = lena_modified.load()
ans_color = ans.load()
print('lena:', lena.mode)

for i in range(0, width):
    for j in range(0, heigth):
        lena_data = lena_color[i, j]
        #print(lena_data)
        lena_modified_data = lena_modified_color[i, j]
        if lena_data != lena_modified_data:
            ans_color[i, j] = lena_modified_data

# ans.show()

ans.save('ans.png')

image = misc.imread('input1.png')
print(image.shape)
mean_pixel = np.mean(image, axis=(0, 1))
print(mean_pixel)