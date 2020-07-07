import cv2
import numpy as np
import time

video_capture = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')# note the lower case
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
out = cv2.VideoWriter('Invisible.mp4',fourcc , 24, (frame_width,frame_height), True)

time.sleep(3)
background = 0

for k in range(45):
    ret,background = video_capture.read()
background = np.flip(background,axis=1)

while(video_capture.isOpened()):
    ret, image = video_capture.read()
    if not ret:
        break
    image = np.flip(image,axis=1)
    #numpy. flip(array, axis) : Reverses the order of array elements along the specified axis, preserving the shape of array.
    
    # Change to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #Hue Saturatio Value It is used for superior vision of ML than RGB

    #Create masks with coordinates to detect the color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    mask_all = cv2.inRange(hsv,low_red,high_red)



    # Tk know about Morphology.....visit https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    """different morphological operations like Erosion, Dilation, Opening, Closing
        They are used for darkening the image lightening the image ,etc."""
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
 
 
    #Hide the blue part away
    mask2 = cv2.bitwise_not(mask_all)
 
    streamA = cv2.bitwise_and(image,image,mask=mask2)

    #Copy the masked area's original part
    streamB = cv2.bitwise_and(background, background, mask = mask_all)
    #To display both the images...Similarly there is also bitwise_or to display either of those masks
    #To know more....visit : https://note.nkmk.me/en/python-opencv-numpy-alpha-blend-mask/
    
 
    #Write the video in the file specified in the previous block
    output = cv2.addWeighted(streamA,1,streamB,1,0)
    out.write(output)
    cv2.imshow("Frame",output)

    key=cv2.waitKey(1)

    if key ==ord('s'):    
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()

"""Red color¶
low_red = np.array([161, 155, 84]) high_red = np.array([179, 255, 255]) red_mask = cv2.inRange(hsv_frame, low_red, high_red) red = cv2.bitwise_and(frame, frame, mask=red_mask)

Blue color
low_blue = np.array([94, 80, 2]) high_blue = np.array([126, 255, 255]) blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue) blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

Green color
low_green = np.array([25, 52, 72]) high_green = np.array([102, 255, 255]) green_mask = cv2.inRange(hsv_frame, low_green, high_green) green = cv2.bitwise_and(frame, frame, mask=green_mask)

Every color except white
low = np.array([0, 42, 0]) high = np.array([179, 255, 255]) mask = cv2.inRange(hsv_frame, low, high) result = cv2.bitwise_and(frame, frame, mask=mask)
"""



    
