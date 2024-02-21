import cv2
import os
import numpy as np

class Deskewer:
    def perform_deskewing(self, image_path, ref_img):
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image is loaded successfully
        if img is None:
            print(f"Error loading image: {image_path}")
            return None

        # Create a SIFT detector object
        sift = cv2.SIFT_create()

        # Detect key points and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # Find the key points and descriptors for the reference image
        ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_img, None)

        # Initialize a Brute-Force Matcher
        bf = cv2.BFMatcher()

        # Match descriptors using KNN
        matches = bf.knnMatch(descriptors, ref_descriptors, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Ensure there are enough good matches to find a homography
        if len(good_matches) >= 4:
            # Extract matched key points
            src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([ref_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate the homography matrix
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Apply the homography to deskew the image
            deskewed_img = cv2.warpPerspective(img, homography, (ref_img.shape[1], ref_img.shape[0]))
            return deskewed_img
        else:
            print(f"Not enough good matches ({len(good_matches)}) to compute homography.")
            return None

    def deskew_image(self, image_path):
        if "front" in image_path.lower():
            ref_img = cv2.imread("deskewing/reference_front.png", cv2.IMREAD_GRAYSCALE)
        else:
            ref_img = cv2.imread("deskewing/reference_back.png", cv2.IMREAD_GRAYSCALE)

        output_image = self.perform_deskewing(image_path, ref_img)
        if output_image is not None:
            output_image_path = "deskewing/deskewed/" + os.path.splitext(os.path.basename(image_path))[0] + ".png"
            cv2.imwrite(output_image_path, output_image)
            return output_image_path
        else:
            return None
