FROM maven:3.8.1-jdk-11 AS fingerprint-wrapper-build

COPY fingerprint-wrapper /usr/src/fingerprint-wrapper
WORKDIR /usr/src/fingerprint-wrapper
RUN mvn clean install

FROM continuumio/miniconda3

COPY msnovelist-env.yml /

RUN /opt/conda/bin/conda env create -f /msnovelist-env.yml

RUN apt-get -qq update && \
	apt-get -qq -y install unzip sqlite &&  \
	wget -q https://bio.informatik.uni-jena.de/repository/dist-release-local/de/unijena/bioinf/ms/sirius/4.4.29/sirius-4.4.29-linux64-headless.zip && \
	unzip -d /usr/local/bin -q sirius-4.4.29-linux64-headless.zip && \
	mkdir -p /root/.sirius

# install yq for better modification of config files
RUN wget -q https://github.com/mikefarah/yq/releases/download/v4.9.6/yq_linux_amd64.tar.gz -O - |\
  tar xz && mv yq_linux_amd64 /usr/bin/yq

COPY . /msnovelist

COPY sirius.sh /usr/local/bin
COPY predict.sh /usr/local/bin

COPY --from=fingerprint-wrapper-build /usr/src/fingerprint-wrapper/target /usr/local/bin/fingerprint-wrapper

RUN echo conda activate msnovelist-env >> /root/.bashrc && \
	echo export COMPUTERNAME=DOCKER >> /root/.bashrc && \
	chmod 777 /msnovelist

WORKDIR /msnovelist