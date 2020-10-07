FROM ubuntu:16.04

RUN apt-get -yqq update && apt-get -y upgrade

RUN apt-get -yqq install python3-pip

RUN apt-get -yqq install htop

COPY . /project

WORKDIR /project

RUN chmod +x boot.sh

RUN pip3 install numpy

RUN pip3 install -r requirements.txt --user

RUN pip3 install gunicorn

ENV FLASK_APP start.py

ENV FLASK_ENV development

EXPOSE 5000

ENTRYPOINT [ "python3" ]

CMD [ "start.py" ]
