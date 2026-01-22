import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def ocr_read_document(image) -> str:
    """Reads an image from the given path and returns extracted text using OCR."""
    # print("config...")
    config=r"--psm 11"
    try:
        text = pytesseract.image_to_string(image,lang='eng',config=config)
        # print("OCR extraction successfull...")
        # print("*"*50,text,"*"*50)
        try:
            text2 = pytesseract.image_to_string(image,lang='hin',config=config)
        except:
            text2=""
        new_text=text+text2
        # print("*"*50,new_text,"*"*50)
        return new_text
    except Exception as e:
        
        return f"Error reading image: {e}"

# from IPython.display import display
# print("Image")
# image_path =r"C:\Users\Abhi9\OneDrive\Documents\Convolve\test\Users\Abhi9\OneDrive\Documents\Convolve\train\173468705_1_pg35.png"
# img=cv2.imread(image_path)
# print("Loaded successfully")
# ocr_read_document(img)
