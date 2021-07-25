# Vaihingen-Semantic-Segmentation

## Outline:

* [Set Up Conda Environment](#Set Up Conda Environment)
* [Data Gathering and Preprocesing](#Data Gathering and Preprocesing)


## Set Up Conda Environment 
```
conda create -f requirements.txt
```

## Data Gathering and Preprocesing 


#### Step 1: ISPRS Vaihingen dataset
The ISPRS Vaihingen dataset can be downloaded with the login details in an 
automated email, after completing this form [form](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html)  via:

ground truth:
```
wget ftp://$username:$password@ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaih
ingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip
```

image data:
```
wget ftp://$username:$password@ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaih
ingen/ISPRS_semantic_labeling_Vaihingen.zip
```

Ones the data has been downloaded move the archives to the `data` folder in project directory and unzip them:
```
unar -d ISPRS_semantic_labeling_Vaihingen.zip
```

```
unar -d ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip
```

#### Step 2: Preprocesing

```
python preprocessing.py
```