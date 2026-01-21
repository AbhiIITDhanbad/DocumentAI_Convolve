from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def ocr_read_document(image) -> str:
    """Reads an image from the given path and returns extracted text using OCR."""
    try:
        text = pytesseract.image_to_string(image)
        # print("OCR extraction successfull...")
        # print("*"*50,text,"*"*50)
        return text
    except Exception as e:
        return f"Error reading image: {e}"
    
# from IPython.display import display

# image_path =r"C:\Users\Abhi9\OneDrive\Documents\Convolve\train\172448470_3_pg15.png"
# ocr_read_document(image_path)
