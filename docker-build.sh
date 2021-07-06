#!/bin/bash

if [[ "$1" != "no-maven" ]]
then
	docker run --rm --name fingerprint-wrapper -v "$(pwd)/fingerprint-wrapper":/usr/src/fingerprint-wrapper -w /usr/src/fingerprint-wrapper maven:3.8.1-jdk-11 mvn clean install
fi

if [[ "$1" != "no-docker" ]]
then
	docker build -t msnovelist .
fi

