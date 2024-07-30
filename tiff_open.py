from PIL import Image
import matplotlib.pyplot as plt

file = "./ASTGTM/ASTGTMV003_N00W061_num.tif"
image = Image.open(file)

plt.imshow(image)
# plt.axis('off')
plt.show()
