# MSNovelist: De novo structure generation from mass spectra
Michael A. Stravs (1), Kai Dührkop (2), Sebastian Böcker (2), Nicola Zamboni (1)

1 Institute of Molecular Systems Biology, ETH Zürich, CH-8092 Zürich, Switzerland

2 Institut für Informatik, Friedrich-Schiller-Universität Jena, D-07743 Jena, Germany

stravs@imsb.biol.ethz.ch

submitted, bioRxiv: https://www.biorxiv.org/content/10.1101/2021.07.06.450875v1

## Build docker

`docker build . -t msnovelist`

## Predict de novo structure

General:

* `docker run -v $DATAFOLDER:/msnovelist-data msnovelist predict.sh SPECTRA SIRIUS_SETTINGS`
* `DATAFOLDER` is a folder that contains at least the spectra to be processed.
* If `SPECTRA` is a **file** within `DATAFOLDER`, it is first processed with SIRIUS. This works with `*.mgf` and `*.ms` (SIRIUS format) files.
* `SIRIUS_SETTINGS` is optional; by default, the settings are `formula -p qtof structure -d ALL_BUT_INSILICO`.
* A `RUNID` (based on the timestamp when running the script) identifies the processing results.
* The SIRIUS results are stored in `DATAFOLDER/sirius-RUNID` and used as input for MSNovelist.
* If `SPECTRA` is a **folder**, it is assumed to be a pre-processed SIRIUS 4.4.29 workspace and used directly as input for MSNovelist
* MSNovelist is then run. 
* If a fingerprint cache exists in `DATAFOLDER/fingerprint_cache.db`, it is used, otherwise a new cache is created at this path
* The used configuration file is deposited as `DATAFOLDER/msnovelist-config-RUNID.yaml`.
* The MSNovelist results are stored in `$DATAFOLDER/results-RUNID/decode-RUNID.csv` and `.pkl`.

Example:
* `docker run -v "$(pwd)/sample-data":/msnovelist-data msnovelist predict.sh 377.mgf`
* (If you don't want to pollute your repository, copy the sample data somewhere else first)

## Info

* Order the results by `score_mod_platt`, descendingly, to get the top candidate (or filter by `rank_score_lim_mod_platt == 1`)
* Multiple spectra (in an MGF file, MS file or SIRIUS project) can be processed in one run, the first column `query` in the result file indicates the spectrum associated with the result




 
