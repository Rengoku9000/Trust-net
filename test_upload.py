import requests
from PIL import Image
import io

url = "http://127.0.0.1:8001/analyze"

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
# Add a small "tampered" block
for x in range(40, 60):
    for y in range(40, 60):
        img.putpixel((x, y), (255, 0, 0)) # still red but let's compress it

img_io = io.BytesIO()
img.save(img_io, 'JPEG', quality=50) # Save with low quality
img_io.seek(0)

# Now simulate an upload
files = {"file": ("test_image.jpg", img_io, "image/jpeg")}
response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
