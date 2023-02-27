# MSNovelist: De novo structure generation from mass spectra
Michael A. Stravs (1), Kai Dührkop (2), Sebastian Böcker (2), Nicola Zamboni (1)

1 Institute of Molecular Systems Biology, ETH Zürich, CH-8092 Zürich, Switzerland

2 Institut für Informatik, Friedrich-Schiller-Universität Jena, D-07743 Jena, Germany

stravs@imsb.biol.ethz.ch

submitted, bioRxiv: https://www.biorxiv.org/content/10.1101/2021.07.06.450875v1

# 27.2.2023 Backend is down

At the moment the CSI:FingerID service backing MSNovelist is **down**. We will need to retrain the model for the new service version, which will unfortunately take a while.

Sorry for the inconvenience.

## Installation and use

*MSNovelist* is provided as a Docker container for end users. This requires a working Docker installation on Windows or Linux; on the other hand, no other dependencies
 are required, the Docker container packages all required software and data.
 
To install Docker on Windows, Linux, or Mac, follow the instructions on https://docs.docker.com/get-docker/. 

Notes:
* Docker on Windows can be installed either with WSL2 or with the HyperV backend (two different ways of running a virtual Linux). Today, WSL2 is typically used 
MSNovelist works with both methods. 
  * If you choose to install HyperV backend, **select Linux containers, not Windows containers**. 
  * If you select HyperV backend, you have to allocate a specified amount of maximal RAM and CPU to Docker; for WSL, the allocation is dynamic.
* For Linux, you can typically use your distribution's package manager.
* We have tested *MSNovelist* on Linux and Windows. There is no reason why it should not work on Mac, however this is currently untested.

After verifying that you have a running Docker installation, pull the latest MSNovelist container:

`docker pull stravsm/msnovelist` 

Alternatively, you can build the container yourself. For this, checkout the Git repository or 
[download the zipped repository](https://github.com/meowcat/MSNovelist/archive/refs/heads/master.zip)
From the repository (the directory containing `Dockerfile`), run `docker build -t msnovelist .`

 No dependencies except  for Docker itself are required. If you build the container on Windows, 
 make sure that the Git repository was checked out with `core.autocrlf=false` (or use the zip file).

MSNovelist can be run as a command-line tool or with a simple Web interface (see below).

## Run web UI

* Ensure you have a running Docker system.
* Open a command line: Powershell (on Windows) or Bash (on Linux)
* Run `docker run -it --init -p 8050:8050 stravsm/msnovelist webui.sh`
  * Or if you want to use your own built image:  `docker run -it --init -p 8050:8050 msnovelist webui.sh`
* Access MSNovelist WebUI on http://localhost:8050
* To terminate the webserver, press Ctrl-C in the shell window.
* To run the server in the background instead: `docker run -d -p 8050:8050 stravsm/msnovelist webui.sh`
  * This server keeps running until you stop it using `docker kill` with the docker ID found with `docker ps`.
  
## Command line interface: Predict de novo structure

General:

* `docker run -v $DATAFOLDER:/msnovelist-data msnovelist predict.sh SPECTRA SIRIUS_SETTINGS`
* `DATAFOLDER` is a folder that contains at least the spectra to be processed.
* If `SPECTRA` is a **file** within `DATAFOLDER`, it is first processed with SIRIUS. This works with `*.mgf` and `*.ms` (SIRIUS format) files.
* `SIRIUS_SETTINGS` is optional; by default, the settings are `formula -p qtof structure -d ALL_BUT_INSILICO`.
* A `RUNID` (based on the timestamp when running the script) identifies the processing results.
* The SIRIUS results are stored in `DATAFOLDER/sirius-RUNID` and used as input for MSNovelist.
* If `SPECTRA` is a **folder**, it is assumed to be a pre-processed SIRIUS 4.4.29 workspace and used directly as input for MSNovelist
  * Note: This is SIRIUS 4.4.29 and not the current SIRIUS version - so you cannot use data processed with the current SIRIUS version.
* MSNovelist is then run. 
* If a fingerprint cache exists in `DATAFOLDER/fingerprint_cache.db`, it is used, otherwise a new cache is created at this path
* The used configuration file is deposited as `DATAFOLDER/msnovelist-config-RUNID.yaml`.
* The MSNovelist results are stored in `$DATAFOLDER/results-RUNID/decode-RUNID.csv` and `.pkl`.


Example:
* Download `377.mgf` from the directory `sample-data` of this repository.
* In the directory with `377.mgf`, run `docker run --init -v "$(pwd)":/msnovelist-data msnovelist predict.sh 377.mgf` 
  * (on Windows in Powershell, use `${pwd}` instead. Alternatively, on either Win or Linux, use the full path.)
* This reproduces the de novo predictions for feature 377 as described in the manuscript. This should work with as little as 4GB of RAM.
* A larger example is `bryophytes.mgf`, the complete bryophyte dataset (576 total spectra). For this, at least 16GB of RAM are suggested. Runtime is approx. 2h on a laptop with 4 cores.

## Info

* Order the results by `score_mod_platt`, descendingly, to get the top candidate (or filter by `rank_score_lim_mod_platt == 1`)
* Multiple spectra (in an MGF file, MS file or SIRIUS project) can be processed in one run, the first column `query` in the result file indicates the spectrum associated with the result

## System requirements

See above: A Docker system able to run Linux Docker containers is required. The Docker container contains all dependencies required to run the software. 
The container was built and tested on Docker 19.03.6, Ubuntu 18.04.4 LTS, with 16 GB RAM; Docker 19.03.8 on Ubuntu 20.04.2 LTS, with 32 GB RAM; 
 Docker Desktop 2.3.0.4 (46911; engine 19.03.12) on Windows 10.0.10942 with 16 GB RAM; and Docker Desktop 4.1.1 (engine v20.10.8) on Windows 10 20H2 (19042.2037).
  The Docker image requires approx. 6.5 GB of disk space.  Build time for the Docker container is up to 20 min. Runtime with a single spectrum is <5 min; 
  for 50 spectra, approx. 30 min on a laptop with 4 cores; / 32GB RAM; for the complete bryophyte dataset, approx. 2:30 h on a machine with 4 cores / 32 GB RAM.



 
