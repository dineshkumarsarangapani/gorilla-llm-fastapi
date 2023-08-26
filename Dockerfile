FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04


#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt install software-properties-common -y 
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install unzip
RUN apt-get -y install python3.10
RUN apt-get -y install python3-pip
RUN apt-get -y install python3.10-distutils
# RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.10 /usr/bin/python3
# RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.10 /usr/bin/python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.10 1
RUN curl https://bootstrap.pypa.io/get-pip.py | python3



# Copy our application code
WORKDIR /var/app

# . Here means current directory.
COPY . .


RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


EXPOSE 80

# Start the app
CMD ["gunicorn", "-b", "0.0.0.0:80","app:app","--workers","1","-k","uvicorn.workers.UvicornWorker", "--timeout","300"]













