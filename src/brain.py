import random
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions
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
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1")
        )

    def extract_structured_data(self, raw_text_list):
        context = "\n".join([item['text'] for item in raw_text_list])

        prompt = f"""
        Extract the following logistics data into a clean JSON format:
        - shipper_name
        - carrier_name
        - bol_number
        - total_weight
        - line_items: (list of description, quantity, weight)

        OCR TEXT:
        {context}
        """

        max_retries = 5
        initial_wait = 12

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=prompt
                )
                return response.text

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (initial_wait * (attempt + 1)) + \
                        random.uniform(0, 3)
                    print(
                        f"⚠️ Rate limit hit. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    return f"AI Error: {str(e)}"
