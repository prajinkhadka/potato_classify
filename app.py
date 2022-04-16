from fastapi import FastAPI, File , HTTPException
import uvicorn
from predict import detect
from PIL import Image
from io import BytesIO
from skimage import io
app = FastAPI()
import os 


# pip install tensorflow==2.7.0
# pip install keras== 2.7.0
# pip install uvicorn==0.13.3
# pip install fastapi==0.63.0
# pip install python-multipart==0.0.5
# pip install scikit-image==0.18.1 

@app.post("/detectpotato/")
async def create_upload_file(file: bytes = File(...)):
    image = Image.open(BytesIO(file))
    image = image.convert("RGB")
    image.save("potato/user.jpg")
    class_label, confidence = detect()
    os.remove('potato/user.jpg')
    return class_label, confidence 

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8501)