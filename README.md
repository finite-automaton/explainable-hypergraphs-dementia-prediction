# Explainable Hypergraph Neural Networks for the Prediction of Dementia Progression
## John Fletcher, MSc Computer Science Dissertation, University of Bath

### Note

The data provided with this submission is artificial and will not
replicate the same results as the dissertation paper. This is necessary
since the NACC data agreement does not allow for sharing of the data.
To reproduce the results, one must apply for access and use data from the May 2023
data freeze.

### Preliminaries

This code has five main files to run:
`synthetic_data.py` `eda.py` `etl.py` `train.py` and  `analysis.py`
The synthetic data file will produce data in a condition which can be run with the code, but is not
real data. Synthetic data is provided in the `semi-supervised-node-classification/data` file, so it's not necessary
to produce new data or run etl for it.
The etl file will generate the necessary hypergraph files from the artifical data.
The eda file will produce eda diagrams for the data.
Please note that some of the diagrams may not formal correctly as the code
was designed for the original, real data.
The train file kicks off the main model, however, it is called via a bash file (more on this to come).
The analysis file produces some post analysis and requires some manual steps after producing output from a model, namely
copying outputted node-edge score csv files, removing incorrect predictions, and then saving them in the appropriate folder.

Moreover, this code has been written primarily to run in a docker image on google cloud hosting to take advantage
of cloud GPUs. You may attempt to install dependencies and run locally, but this is likely to 
result in errors if the dependency management isn't perfect.

The instructions below require you to build a docker file (This can be run locally, and with fake data having many less row,
it runs reasonably quickly). This is quite a time consuming process and requires sufficient disk space.

### Instructions to run code
**Note: You must have a docker client running on your machine**

1. Copy the code onto your machine
2. (optional) *Synthetic data is already provided in semi-supervised-node-classification > data* run synthetic_data with your desired number of rows specified, this will create a file in `hypergraph_data/raw_data` you must delete all of the files except `final_vals_missing_codes.csv` in that directory if repeating that process 
3. (optional) run eda.py to generate diagrams, if desired
3. In terminal, navigate to the `semi-supervised-node-classification` folder
4. run `docker build -t local:jfdissertation` (this will create a docker build and may take some time)
5. run `docker run -it local:jfdissertation bash` (this will run a local docker container with the work and open a bash terminal)
6. (if using provided synthetic data) inside the docker container, run the command `bash nacc.sh 4_visits_LESS_300_hg_input 50 10 0 1`

The bash command takes the following structure:
`bash nacc.sh <data folder> <number of total epochs> <number of warmup epochs> <ehnn only == 1> <number of runs>`

If you want to run on synthetic data, copy that folder from hypergraph_data to semi-supervised-node-classification/data before building the docker image
or copy it over. It must contain valid nacc.content and nacc.edges files