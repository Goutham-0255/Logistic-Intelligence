import os
from src.preprocess import DocumentPreProcessor
from src.extractor import DataExtractor
from src.brain import DocumentBrain


def main():
    # 1. Setup Paths
    # Ensure these match your actual filenames in data/raw/
    input_file = "data/raw/test_invoice.jpg"
    cleaned_file = "data/processed/cleaned_invoice.jpg"

    # Create folder if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    print("🚀 Starting Intelligent Document Pipeline...")

    # --- PHASE 1: PRE-PROCESSING (Computer Vision) ---
    print("\n🎨 Step 1: Cleaning and Deskewing image...")
    try:
        processor = DocumentPreProcessor(input_file)
        processor.rescale_image()
        processor.save_processed(cleaned_file)
    except Exception as e:
        print(f"❌ Error in Pre-processing: {e}")
        return

    # --- PHASE 2: EXTRACTION (OCR) ---
    print("\n🔍 Step 2: Extracting raw text with EasyOCR...")
    try:
        extractor = DataExtractor()
        raw_ocr_results = extractor.extract_text(cleaned_file)

        # Optional: Print raw results to see what EasyOCR found
        print(f"Found {len(raw_ocr_results)} text snippets.")
    except Exception as e:
        print(f"❌ Error in OCR Extraction: {e}")
        return

    # --- PHASE 3: INTELLIGENCE (LLM) ---
    print("\n🧠 Step 3: Structuring data with Gemini AI...")
    try:
        brain = DocumentBrain()
        structured_json = brain.extract_structured_data(raw_ocr_results)
    except Exception as e:
        print(f"❌ Error in AI Structuring: {e}")
        return

    # --- FINAL OUTPUT ---
    print("\n" + "="*40)
    print("🎯 FINAL STRUCTURED LOGISTICS DATA")
    print("="*40)
    print(structured_json)
    print("="*40)
    print("\n✅ Pipeline Complete.")


if __name__ == "__main__":
    main()
