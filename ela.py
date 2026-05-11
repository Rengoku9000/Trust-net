from PIL import Image, ImageChops, ImageEnhance
import io
import numpy as np

def perform_ela(file_bytes: bytes) -> int:
    """
    Performs Error Level Analysis (ELA) on an image to detect pixel-level tampering.
    Returns a score from 0 to 100.
    """
    try:
        # Attempt to load the file as an image
        original = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # Resave it at a known quality (e.g., 90)
        resaved_io = io.BytesIO()
        original.save(resaved_io, "JPEG", quality=90)
        resaved = Image.open(resaved_io)
        
        # Calculate the absolute difference between the original and resaved image
        diff = ImageChops.difference(original, resaved)
        
        # Convert to numpy array for statistical analysis
        diff_array = np.array(diff)
        
        # Sum of differences across RGB channels for each pixel
        diff_sum = np.sum(diff_array, axis=2)
        
        # A simple heuristic: High standard deviation in differences often indicates
        # localized tampering, because tampering introduces new edges/compression artifacts
        # that stand out from the rest of the image.
        std_dev = np.std(diff_sum)
        
        # Normalize to a 0-100 score.
        # Empirically, a std_dev > 15 is highly suspicious for standard images.
        score = min(100, max(0, int((std_dev / 15.0) * 100)))
        
        return score
    except Exception as e:
        # This will trigger if the file is a PDF or unsupported format.
        # For now, we return 0 for non-images in the ELA layer.
        print(f"ELA skipped or failed: {e}")
        return 0
