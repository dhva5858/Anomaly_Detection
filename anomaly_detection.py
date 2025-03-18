import cv2
import numpy as np
import os

defect_free_path = r"C:\Users\dpvas\Downloads\data\data\Golden images\golden_image_1.png"
defective_path = r"C:\Users\dpvas\Downloads\data\data\pcb1_defect_images\defect_image_1_.png"

defect_free = cv2.imread(defect_free_path, cv2.IMREAD_GRAYSCALE)
defective = cv2.imread(defective_path, cv2.IMREAD_GRAYSCALE)

#Align the Images (Handle Rotation)
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(defect_free, None)
keypoints2, descriptors2 = orb.detectAndCompute(defective, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #Match keypoints using the BFMatcher
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) #Extract matched keypoints
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0) # Estimate homography to align the images
aligned_defective = cv2.warpPerspective(defective, matrix, (defect_free.shape[1], defect_free.shape[0]))

difference = cv2.absdiff(defect_free, aligned_defective) #Compute the Difference

_, thresh = cv2.threshold(difference, 100, 255, cv2.THRESH_BINARY) #Threshold the Difference

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#Find Contours of the Differences

defective_color = cv2.imread(defective_path)

#Draw red boxes around the detected anomalies
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w * h > 100:  # Minimum area threshold
        cv2.rectangle(defective_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
output_dir = r"C:\Users\dpvas\Downloads\data\outputimage"
output_path = os.path.join(output_dir, "pcb1_defect _highlighted.png")
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(output_path, defective_color)