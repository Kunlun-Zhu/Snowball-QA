<div align="center">
<div align="center">

<h1>Snowball-QA</h1>

**A toolkit for Chinese question answering pair data augmentation**

<p align="center">
  <a href="#overview">Overview</a>  • <a href="#install">Installation</a> • <a href="#usage">Usage</a> • <a href="#results">Results</a> • <a href="#todo">ToDo</a> 
<br>
</p>

<p align="center">



<a href="http://www.apache.org/licenses/">
    <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/BMTrain">
</a>

</p>

</div>
</div>

## Overview

Our toolkit aiming for generate augmentation QA data from document data

### Architecture


```
├── answer_extractor
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── question_generator
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
└── qa_filter
    ├── __init__.py
    └── pert_mrc.py
```

### Data
We pre-trained our answer extractor and question generator with the following dataset 
- Dureader 
- CMRC2018 
- ChineseSquad 

### Model
We chose 
- [Chinese-Bert-wwm](https://github.com/ymcui/Chinese-BERT-wwm) as our answer-extractor model   
- [Mengzi](https://github.com/Langboat/Mengzi) as our question generation model  
- [PERT](https://github.com/ymcui/PERT) as our QA=filter model  

### pipleline

## Installation

Clone the Snowball-QA from github:
```bash
git clone https://github.com/Kunlun-Zhu/Snowball-QA.git
cd Snowball-QA
pip install -r requirements.txt
```
## Usage
### Step 1: download model&data
You can download our model checkpoint and training data in Ali-cloud 

### Step2: train answer extractor or load the model from checkpoint
```bash
sh run_train_ae.sh
```
### Step3: train the question generator or load the model from checkpoint
```bash
sh run_train_qg.sh
```
### Step4: get answer extration results & question generation results
```bash
sh run_predict_ae.sh
sh run_predict_qg.sh
```
### Step5: run QA filter to get the final augmentation data
```bash
sh run_filter.sh
```
### Try your own data
You could try your own data with the format as the file [demo.jsonl]()   
And change the directory to the file in [pipeline.sh]()



## Results

## ToDo
