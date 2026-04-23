import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions  # Added for version control
from pydantic import BaseModel
from typing import Optional, List
load_dotenv()


class LineItem(BaseModel):
    description: str
    quantity: Optional[str]
    weight: Optional[str]


class DocumentBrain:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # We force 'v1' to avoid the 404/v1beta errors
        self.client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1")
        )

    def extract_structured_data(self, raw_text_list):
        context = "\n".join([item['text'] for item in raw_text_list])

        prompt = f"""
        Extract logistics data into JSON. 
        Fields: shipper_name, carrier_name, bol_number, line_items.
        
        OCR TEXT:
        {context}
        """

        print("📡 Sending to Gemini...")
        try:
            # Use just the string name. The SDK handles the 'models/' prefix.
            # If 1.5-flash fails, 'gemini-1.5-flash-latest' is the foolproof backup.
            response = self.client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            # If it still fails, let's catch exactly what the model name was
            return f"AI Error: {str(e)}"
