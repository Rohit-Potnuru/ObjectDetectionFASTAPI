from typing import List

import io
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,  Response
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse


import os
from PIL import Image
import torch

import Config
import pickle
from ProcessImages import processingImages
from CustomUtils import deleteImages




# Importing the StringIO module.


app = FastAPI()
app.mount("/output", StaticFiles(directory="images/output"), name="static")
app.mount("/input", StaticFiles(directory="images/input"), name="static")

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    

@app.post("/uploadfiles/")
# async def create_upload_files(files: UploadFile = File(...)):
async def create_upload_files(files: List[UploadFile] = File(...)):
    # print("hello Tanvi")
    
    input_imgs = []
    for file in files:
        
        file_name = os.path.join(Config.INPUT_DIR, file.filename.replace(" ", "-"))
        with open(file_name,'wb+') as f:
            f.write(file.file.read())
            f.close()
        input_imgs.append(file.filename)
    
    output_imgs = processingImages(input_imgs) # Process image and gives output images
    
    json = {
        "input": {"input": input_imgs},
        "output": {"output": output_imgs}
    }

    return json

@app.get("/")  # sends a html page for uploadingmimage files
async def main():
    content = """
<body>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)




























# @app.post("/don/")
# async def root():
#     return {"message": "Hello World"}

# @app.post("/files/")
# async def create_files(files: List[bytes] = File(...)):
#     return {"file_sizes": [len(file) for file in files]}


# # @app.post("/uploadfiles/")
# # async def create_upload_files(files: List[UploadFile] = File(...)):
# #     print("hello Tanvi")
# #     return {"filenames": [file.filename for file in files]}

# @app.post("/vector_image")
# def image_endpoint(*, vector):
#     # Returns a cv2 image array from the document vector
#     cv2img = cv2.imread("/images/input/0ce9ce9f-c0ec-49b2-aaea-8bb39453959c.jpg",'r')
#     return StreamingResponse(io.BytesIO(cv2img.tobytes()), media_type="image/jpg")

# @app.get("/vector_image1")
# async def main():
#     # Returns a cv2 image array from the document vector]
#     return FileResponse("images/input/0ce9ce9f-c0ec-49b2-aaea-8bb39453959c.jpg")
