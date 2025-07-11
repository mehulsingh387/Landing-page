import cv2
import numpy as np
import easyocr

def test_ocr():
    print("🚀 Starting OCR Test...")
    
    # Create a test image with text
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(img, "TEST123", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save the test image
    cv2.imwrite("test_ocr_input.png", img)
    print("✅ Created test image: test_ocr_input.png")
    
    # Initialize OCR
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'])
    
    # Test 1: Read directly from the image
    print("\n--- Test 1: Reading from image ---")
    results = reader.readtext("test_ocr_input.png")
    print("Results from image file:", results)
    
    # Test 2: Read from numpy array
    print("\n--- Test 2: Reading from numpy array ---")
    results = reader.readtext(img)
    print("Results from numpy array:", results)
    
    # Test 3: Try with different preprocessing
    print("\n--- Test 3: With thresholding ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("test_ocr_thresh.png", thresh)
    results = reader.readtext(thresh)
    print("Results from thresholded image:", results)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_ocr()
    