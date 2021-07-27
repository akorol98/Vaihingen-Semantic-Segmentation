# Vaihingen-Semantic-Segmentation
![](figures/metrics.png)
## Outline:

* [Set Up Conda Environment](#Set-Up-Conda-Environment)

* [Data Gathering and Preprocesing](#Data-Gathering-and-Preprocesing)

* [Model training](#Model-training)


# Set Up Conda Environment 
```
conda create -f environment.yml
```

# Data Gathering and Preprocesing 


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

# Model training

### Baseline Model

To train a baseline model, run:
```
python train.py baseline
```

### Weakly Supervised Model

The weakly supervised training is performed in three steps:

#### Step 1: Train baseline
Train the baseline model (if still not trained). For that see the Baseline Model section.
#### Step 2: Generate masks for weakly training
Generate the masks using the trained baseline model. For this run from the main challenge directory
```
python gen_masks.py
```
This will generate the masks which are corrected by image level labels: if on the image level some tag is not present, all the corresponding pixel level tags of this type are deleted.

#### Step 3: Run the actual training of the weakly supervised model
 
```
python train.py weakly_supervised
```
The idea behind this model is that one uses the masks generated by the baseline model to use this information as a support to training on N1 + N2(weakly) combined sample.
