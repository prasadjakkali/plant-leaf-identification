# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
import cv2

filename = "gt.jpg"
test_img_path = '..\\mobile captures\\' + filename
main_img = cv2.imread(test_img_path)
plt.imshow(main_img)


#vv.v.imp
plt.title('matplotlib.pyplot.imshow() function Example',fontweight ="bold")
plt.show()
