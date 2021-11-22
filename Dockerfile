FROM maven:3.8.1-jdk-11 AS fingerprint-wrapper-build

COPY fingerprint-wrapper /usr/src/fingerprint-wrapper
WORKDIR /usr/src/fingerprint-wrapper
RUN mvn clean install

FROM continuumio/miniconda3:4.10.3

COPY msnovelist-env.yml /

RUN /opt/conda/bin/conda config --set remote_connect_timeout_secs 40 && \
    /opt/conda/bin/conda config --set remote_read_timeout_secs 100 && \
    /opt/conda/bin/conda env create -f /msnovelist-env.yml

# install yq for better modification of config files
RUN wget -q https://github.com/mikefarah/yq/releases/download/v4.9.6/yq_linux_amd64.tar.gz -O - |\
  tar xz && mv yq_linux_amd64 /usr/bin/yq

COPY . /msnovelist

#RUN apt-get -qq update && \
#	apt-get -qq -y install unzip sqlite &&  \
#	unzip -d /usr/local/bin -q /msnovelist/sirius_bin/sirius-linux64-headless-4.4.29.zip && \
#	mkdir -p /root/.sirius && \
#	rm -r /msnovelist/sirius_bin

RUN apt-get -qq update && \
 	apt-get -qq -y install unzip sqlite &&  \
 	wget -q https://github.com/boecker-lab/sirius/releases/download/post-4.0.1/sirius-linux64-headless-4.4.29.zip && \
 	unzip -d /usr/local/bin -q sirius-linux64-headless-4.4.29.zip && \
 	mkdir -p /root/.sirius



COPY *.sh /usr/local/bin/

COPY --from=fingerprint-wrapper-build /usr/src/fingerprint-wrapper/target /usr/local/bin/fingerprint-wrapper

RUN echo conda activate msnovelist-env >> /root/.bashrc && \
	echo export COMPUTERNAME=DOCKER >> /root/.bashrc && \
	chmod 777 /msnovelist && \
	mkdir /msnovelist-data && \
	chmod 777 /msnovelist-data

WORKDIR /msnovelist
