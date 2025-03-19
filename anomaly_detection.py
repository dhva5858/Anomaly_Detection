import cv2
import numpy as np
import os

def load_image(image_path, grayscale=True):
    """Load an image in grayscale or color mode."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(image_path, flag)

def align_images(reference_img, target_img):
    """Align the target image to the reference image using ORB feature matching."""
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(reference_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target_img, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(target_img, matrix, (reference_img.shape[1], reference_img.shape[0]))
    
    return aligned_img

def detect_anomalies(reference_img, aligned_img):
    """Detect anomalies by computing the absolute difference and thresholding."""
    difference = cv2.absdiff(reference_img, aligned_img)
    _, thresh = cv2.threshold(difference, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def highlight_defects(original_img, contours, min_area=100):
    """Draw bounding boxes around detected anomalies."""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return original_img

def save_image(image, output_path):
    """Save the processed image to the specified path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def main():
    """Main function to execute the anomaly detection pipeline."""
    defect_free_path = r"C:\Users\dpvas\Downloads\data\data\Golden images\golden_image_1.png"
    defective_path = r"C:\Users\dpvas\Downloads\data\data\pcb1_defect_images\defect_image_1_.png"
    output_path = r"C:\Users\dpvas\Downloads\data\outputimage\pcb1_defect_highlighted.png"
    
    defect_free = load_image(defect_free_path)
    defective = load_image(defective_path)
    defective_color = load_image(defective_path, grayscale=False)
    
    aligned_defective = align_images(defect_free, defective)
    contours = detect_anomalies(defect_free, aligned_defective)
    result_img = highlight_defects(defective_color, contours)
    
    save_image(result_img, output_path)
    print(f"Processed image saved at: {output_path}")

if __name__ == "__main__":
    main()
