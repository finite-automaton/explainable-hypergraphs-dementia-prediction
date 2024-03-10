FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310

ADD requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir -p /dmrehnn
WORKDIR /dmrehnn
COPY . .

CMD ["/bin/bash"]
