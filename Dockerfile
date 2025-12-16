FROM python:3.10-slim
WORKDIR /
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY rp_handler.py /rp_handler.py
CMD ["python3", "-u", "/rp_handler.py"]
