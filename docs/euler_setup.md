## Process to set up on Euler

* Checkout git repo on login node. Seemingly doesn't work on job node.
* Build singularity image on job node. Freaks out on login node: 
    ```
    SCRATCH_PATH=/cluster/scratch/$(id -un)
    mkdir -p $SCRATCH_PATH/singularity_cache
    mkdir -p $SCRATCH_PATH/MSNovelist-image
    SINGULARITY_CACHEDIR=$SCRATCH_PATH/singularity_cache singularity build \
            $SCRATCH_PATH/MSNovelist-image/msnovelist.sif docker://stravsm/msnovelist6
    ```
* Build AWS image on job node. Set `SCRATCH_PATH` first.
    ```
    SINGULARITY_CACHEDIR=$SCRATCH_PATH/singularity_cache singularity build $SCRATCH_PATH/aws.sif docker://public.ecr.aws/aws-cli/aws-cli
    ```

* Run AWS image and download data
    ```
    singularity shell --bind $TMPDIR:/$HOME $SCRATCH_PATH/aws.sif
    # in singularity shell:
    aws configure
    # set up the access data
    mkdir data
    aws s3 cp --recursive s3://sirius-novelist/dataset-s6-202311 data
    ```

* Your data is now at `$TMPDIR/data`. Exit the container and copy to scratch:
    ```
    cp -r $TMPDIR/data /cluster/scratch/stravsm/MSNovelist-data
    ```

* Set up `.env` such that it looks more or less like this:
    ```
    DATA_LOC=/cluster/scratch/username/MSNovelist-data
    SIF_LOC=/cluster/scratch/username/msnovelist.sif
    CODE_LOC=/cluster/home/username/MSNovelist-private
    RESULTS_LOC=/cluster/scratch/username/MSNovelist-results
    ```

* To train, run: 
    ```
    sbatch run_singularity.sh
    ```

## Upload trained weights to S3

* On a job node, run AWS image and upload weights
    ```
    singularity shell --bind $SCRATCH_PATH:/data --bind $TMPDIR:/$HOME $SCRATCH_PATH/aws.sif
    # in singularity shell:
    aws configure
    # go to training results
    cd /data/MSNovelist-results/weights
    # choose a weights set to upload, and copy:

    aws s3 cp --recursive m-36719628-msnovelist-sirius6-weights s3://sirius-novelist/m-36651659
    ```
