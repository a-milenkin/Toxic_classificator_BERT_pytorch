# python:alpine is 3.{latest}
FROM python

RUN pip install flask && pip install numpy && pip install  torch&& pip install   transformers && pip install  nltk && pip install pandas
#RUN pip install python-daemon

COPY src /src/
#COPY /scr/model /src/model

EXPOSE 5000

ENTRYPOINT ["/bin/bash", "/src/start.sh" ]
