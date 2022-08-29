<div align="center">
<div align="center">

<h1>Snowball-QA</h1>

**A toolkit for Chinese question answering pair data augmentation**

<p align="center">
  <a href="#overview">Overview</a>  • <a href="#install">Installation</a> • <a href="#usage">Usage</a> • <a href="#results">Results</a> 
<br>
</p>

<p align="center">



<a href="https://github.com/OpenBMB/BMTrain/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/BMTrain">
</a>

</p>

</div>
</div>

## Overview

### Data
We pre-trained our answer extractor and question generator with the following dataset 
- Dureader 
- CMRC2018 
- ChineseSquad 

### Model
We chose [Chinese-Bert-wwm](https://github.com/ymcui/Chinese-BERT-wwm) as our answer-extractor model   
We chose [Mengzi](https://github.com/Langboat/Mengzi) as our question generation model  
We chose [PERT](https://github.com/ymcui/PERT) as our QA=filter model  

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
You can get our model checkpoint and data in Ali-cloud 

### Step2: train answer extractor or load the model from checkpoint
```bash
sh run_train_ae.sh
```
### Step3: train the question generator or load the model from checkpoint
```bash
sh run_
```
## Results