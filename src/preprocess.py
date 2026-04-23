import cv2
import numpy as np
import os


class DocumentPreProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not find image at {image_path}")

    def rescale_image(self, width=1000):
        """
        Standardizes image size. OCR engines perform better 
        when text size is consistent.
        """
        height = int(self.image.shape[0] * (width / self.image.shape[1]))
        self.image = cv2.resize(
            self.image, (width, height), interpolation=cv2.INTER_AREA)
        return self.image

    def fix_tilt(self, binary_img):
        """
        Detects the skew angle of the text and rotates the image back to 0 degrees.
        Essential for tabular data extraction in logistics documents.
        """
        # Invert colors: Hough lines work better on white text on black background
        inverted_img = cv2.bitwise_not(binary_img)

        # Find all non-zero pixels (the text)
        coords = np.column_stack(np.where(inverted_img > 0))

        # Find the minimum area rectangle that encloses the text coordinates
        angle = cv2.minAreaRect(coords)[-1]

        # OpenCV angle logic: adjust the angle so it rotates correctly
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Perform the rotation
        (h, w) = binary_img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary_img, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        if angle != 0:
            print(f"📐 Skew detection: {angle:.2f} degrees.")

        return rotated

    def apply_cleaning(self):
        """
        The 3-Pillar Pipeline:
        1. Grayscale: Simplify color data.
        2. Bilateral Filter: Remove noise while keeping text edges sharp.
        3. Adaptive Threshold: Fix uneven lighting/shadows.
        """
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 2. Denoising (Bilateral Filter)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. Binarization (Adaptive Thresholding)
        binary_map = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 4. Deskewing (Fixing Tilt)
        final_img = self.fix_tilt(binary_map)

        return final_img

    def save_processed(self, output_path):
        """Executes the pipeline and saves the file."""
        processed_img = self.apply_cleaning()
        cv2.imwrite(output_path, processed_img)
        print(f"✅ Processed image saved to: {output_path}")
