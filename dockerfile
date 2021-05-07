FROM python:3.7

RUN pip install fastapi uvicorn torch Pillow aiofiles torchvision
RUN pip install python-multipart


WORKDIR /app
COPY . /app
CMD ["uvicorn", "main:app"]
EXPOSE 8000
