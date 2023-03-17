
FROM python:3.8.10-slim

LABEL maintainer="petros.tamvakis@athenarc.gr"

# RUN apt update -y
# RUN apt install -y python
# RUN apt install pip 
# RUN pip install python-dev-tools
# RUN python-dev 

WORKDIR /app

# RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx
COPY requirements.txt /app/requirements.txt

RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt
RUN venv/bin/pip install gunicorn


# RUN apt-get update && apt-get install -y python3-opencv
# RUN venv/bin/pip install opencv-python==4.5.3.56


COPY . /app

# COPY ./mobilenetV2_plantvil /app/mobilenetV2_plantvil
# COPY ./saliency_imgs /app/saliency_imgs
# COPY ./app/templates /app/templates
# COPY ./app/utils /app/utils

# COPY microblog.py config.py boot.sh ./

RUN chmod +x boot.sh

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]

# CMD [ "app/main.py" ]