# Russian License Plate Detection 
For this project, I focused on Russian License Plate Detection, employing a combination of Haar Cascade classifiers, image preprocessing, and Optical Character Recognition (OCR) with Tesseract. The Haar Cascade classifier was fine-tuned with parameters like scaleFactor=1.2 and minNeighbors=3 to enhance detection accuracy, particularly for plates at varying distances and with fewer neighboring features. Preprocessing techniques, including grayscale conversion, histogram equalization, resizing, and noise reduction, were crucial in preparing the images for OCR. Tesseract OCR, configured with specific character whitelists and tested with various Page Segmentation Modes, extracted text, which was then validated using regular expressions to match the structure of Russian plates.







