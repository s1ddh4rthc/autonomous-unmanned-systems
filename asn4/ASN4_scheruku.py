import cv2
import numpy as np
from matplotlib import pyplot as plt

# Converting the image to grayscale

cv2.imshow("Grayscale", cv2.imread('/Users/siddharthcherukupalli/repos/autonomous-unmanned-systems/asn4/croppedBarrel.png', 0))
cv2.waitKey(1)

vid = cv2.VideoCapture('/Users/siddharthcherukupalli/repos/autonomous-unmanned-systems/asn4/Vid.mp4')
img = cv2.imread('croppedBarrel.png')

writer = cv2.VideoWriter('BarrelDetector.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))


while vid.isOpened():
    ret, frame = vid.read()
    cv2.imshow('Frame', frame)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(40)

    template = cv2.imread('croppedBarrel.png', 0)
    template = cv2.resize(template, (50, 75))
    w, h = template.shape[::-1]

    method = cv2.TM_SQDIFF
    
    # Apply template Matching
    res = cv2.matchTemplate(grayFrame, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, 255, 2)
    cv2.imshow('Image', frame)
    
    writer.write(frame)
    cv2.waitKey(50)

writer.release()

