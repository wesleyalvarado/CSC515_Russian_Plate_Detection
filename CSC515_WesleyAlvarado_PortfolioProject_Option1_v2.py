import cv2
import numpy as np
import pytesseract
import re 

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1/bin/tesseract'

# Load Haar Cascade classifier for Russian license plates
cascade_path = 'haarcascade_license_plate_rus_16stages.xml'
plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    raise IOError(f"Unable to load the cascade classifier from {cascade_path}")

allowed_characters = "ABbEKkMHOPCTyYX0123456789"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def detect_license_plate(gray_image):
    plates = plate_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print(f"Detected {len(plates)} plates")
    return plates

def correct_orientation(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None:
        angles = [theta * 180 / np.pi - 90 for rho, theta in lines[:, 0]]
        median_angle = np.median(angles)
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    return plate_img

def preprocess_plate_image(plate_img):
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray_plate = cv2.equalizeHist(gray_plate)
    resized_plate = cv2.resize(gray_plate, (400, 100))
    filtered = cv2.fastNlMeansDenoising(resized_plate, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

def recognize_characters(plate_img):
    best_text = ""
    best_psm = 6
    for psm in [6, 9, 11]:
        print(f'PSM: {psm}')
        config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist={allowed_characters}'
        text = pytesseract.image_to_string(plate_img, config=config).strip()
        print(f"Detected text (PSM {psm}): {text}")

        # Update best text if current text is longer
        if len(text) > len(best_text):
            best_text = text
            best_psm = psm

    print(f"Best PSM: {best_psm}")

    # Remove unwanted characters (spaces, newlines) using regex
    cleaned_text = re.sub(r'\s+', '', best_text)  # Remove all whitespace, including newlines
    print(f"Cleaned text: {cleaned_text}")

    # Define the expected pattern for the license plate
    pattern = r'^([ABEKMHOPCTYX][0-9]{3}[ABEKMHOPCTYX]{2})([0-9]{2,3})?$'

    # Match the cleaned text against the pattern
    match = re.match(pattern, cleaned_text)
    if match:
        # Extract the valid part of the plate number and region code
        valid_text = match.group(1)
        region_code = match.group(2) if match.group(2) else ''
    else:
        # Handle cases where the cleaned text doesn't fully match the pattern
        valid_text = cleaned_text
        # Extract and remove region code if present
        region_match = re.search(r'(\d{2,3})$', valid_text)
        if region_match:
            region_code = region_match.group(1)
            valid_text = re.sub(r'(\d{2,3})$', '', valid_text)
        else:
            region_code = ''
    
    # Replace '0' with 'O' only if it should be a letter
    corrected_text = ''
    for i, char in enumerate(valid_text):
        # If the character is '0' and should be a letter, replace with 'O'
        if char == '0' and i in [5,6]:  # Adjust positions if needed
            corrected_text += 'O'
        else:
            corrected_text += char

    # Ensure there are no letters after the region code
    plate_number = f"{corrected_text}{region_code}"
    plate_number = re.sub(r'[A-Z]$', '', plate_number)  # Remove any trailing letters
    print(f"Final plate number: {plate_number}")

    return plate_number

def main():
    image_paths = [
        'jersey_plate.jpg',
        'russian_plate5.jpg',
        'russian_plate6.jpg'
    ]
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue
        
        gray = preprocess_image(image)
        plates = detect_license_plate(gray)

        for idx, (x, y, w, h) in enumerate(plates):
            plate_img = image[y:y+h, x:x+w]
            plate_img = correct_orientation(plate_img)
            
            preprocessed_plate = preprocess_plate_image(plate_img)
            
            plate_text = recognize_characters(preprocessed_plate)
            if plate_text:
                print(f"Detected text for plate {idx}: {plate_text}")
                
                # Draw a red rectangle around the detected license plate
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Put the detected text on the image
                cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print(f"No valid text detected for plate {idx}")

        detected_output_path = f'detected_{image_path}'
        cv2.imwrite(detected_output_path, image)
        print(f"Saved detected plates to {detected_output_path}")

        cv2.imshow('Detected Plates', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
