import numpy as np
import os
import cv2
import matplotlib.pyplot as plt




img_path = './0000-0007L_1000.jpg'
image = cv2.imread(img_path)
frame = image[:496, :image.shape[1]-768, :]
image_p2 = image[:496, image.shape[1]-768+128:image.shape[1]-128, :]


hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 

lower_green = np.array([50, 100, 100])
upper_green = np.array([77, 255, 255])

green_mask = cv2.inRange(hsv, lower_green, upper_green)

value = np.sum(green_mask, axis=1)
i = 40
while i<496:
    if value[i] < 2000:
        green_mask[i,:] = 0
    if value[i] > 100000 and value[i+1] > 100000:
        green_mask[i-1,:] = 255
        green_mask[i,:] = 255
        green_mask[i+1,:] = 255
        green_mask[i+2,:] = 255
    i = i+1

green_res = cv2.bitwise_and(frame, frame, mask = green_mask)
dst_inpaint = cv2.inpaint(frame,green_mask,3,cv2.INPAINT_NS)
cv2.imwrite(os.path.join('./','output_3.jpg'), dst_inpaint)
cv2.imwrite(os.path.join('./','output_4.jpg'), green_mask)
cv2.imwrite(os.path.join('./','output_5.jpg'), green_res)
frame = frame[:,:,::-1]
green_res = green_res[:,:,::-1]
dst_inpaint = dst_inpaint[:,:,::-1]


plt.figure(figsize=(14,12))
plt.subplot(2,2,1),plt.title('original_image'), plt.imshow(frame)
plt.subplot(2,2,2),plt.title('green things'), plt.imshow(green_res)
plt.subplot(2,2,3),plt.title('green masks'), plt.imshow(green_mask, cmap= 'gray')
plt.subplot(2,2,4),plt.title('inpainted image'), plt.imshow(dst_inpaint)

plt.show()