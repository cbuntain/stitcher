import cv2
import numpy as np

def showImage(img1, 
    pts=None,
    scale=None,
    title="Image", timeout=1000):

    # # Show me the tracked points!
    if ( len(img1.shape) == 3 ):
        composite_image = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8)
    else:
        composite_image = np.zeros((img1.shape[0], img1.shape[1]), np.uint8)

    composite_image[:,0:img1.shape[1]] += img1

    if ( pts != None ):
        for pt in pts:
            cv2.circle(composite_image, (pt[0], pt[1]), 5, 255)

    if ( scale != None ):
        fx = scale[0]
        fy = scale[1]

        composite_image = cv2.resize(composite_image, (0,0), fx=fx, fy=fy) 

    cv2.imshow(title, composite_image);
    cv2.waitKey(timeout)

def showComposite(img1, img2, 
    pts1=None, pts2=None, 
    scale=None,
    title="Image", timeout=1000):

    # # Show me the tracked points!
    composite_image = np.zeros((img1.shape[0], img1.shape[1]*2), np.uint8)
    composite_image[:,0:img1.shape[1]] += img1
    composite_image[:,img1.shape[1]:img2.shape[1]*2] += img2

    if ( pts1 != None ):
        for pt in pts1:
            cv2.circle(composite_image, (pt[0], pt[1]), 5, 255)

    if ( pts2 != None ):
        for pt in pts2:
            x = int(pt[0]) + img1.shape[1]
            y = int(pt[1])
            cv2.circle(composite_image, (x, y), 5, 255)

    if ( scale != None ):
        fx = scale[0]
        fy = scale[1]

        composite_image = cv2.resize(composite_image, (0,0), fx=fx, fy=fy) 

    cv2.imshow(title, composite_image);
    cv2.waitKey(timeout)