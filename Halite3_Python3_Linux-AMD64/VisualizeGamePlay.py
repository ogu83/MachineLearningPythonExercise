import cv2
import numpy as np

for i in range(2,400):
    d = np.load(f"gameplay/{i}.npy")
    print(d)

    cv2.imshow("", cv2.resize(d, (0,0), fx=25, fy=25))
    cv2.waitKey(25)    
