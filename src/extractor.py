import easyocr


class DataExtractor:
    def __init__(self):
        # Initialize the reader for English (you can add 'hi', 'fr', etc. later)
        # gpu=False is safer for local dev, change to True if you have an NVIDIA GPU
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image_path):
        """
        Reads the image and returns a list of dictionaries 
        containing the text and its location.
        """
        print(f"🔍 Extracting text from {image_path}...")

        # This is the core OCR command
        results = self.reader.readtext(image_path)

        structured_data = []
        for (bbox, text, prob) in results:
            # We only keep text with a decent confidence score
            if prob > 0.40:
                structured_data.append({
                    "text": text,
                    "location": bbox,
                    "confidence": round(prob, 2)
                })

        return structured_data

    def get_simple_text(self, results):
        """Helper to get a clean string of all detected text."""
        return " ".join([item['text'] for item in results])
