import random
import time
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
    prompt = f"Extract logistics data into JSON: {context}"

    # Configuration for "Never Fail" logic
    max_retries = 5
    initial_wait = 12  # Free tier resets every 60s, so 12s is a safe step

    for attempt in range(max_retries):
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )
            return response.text

        except Exception as e:
            # Check if the error is a Rate Limit (429)
            if "429" in str(e) and attempt < max_retries - 1:
                # Exponential Backoff: wait 12s, then 24s, then 48s...
                wait_time = (initial_wait * (attempt + 1)) + \
                    random.uniform(0, 3)
                print(
                    f"⚠️ Rate limit hit. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                # If it's a different error or we're out of retries, give up
                return f"AI Error: {str(e)}"
