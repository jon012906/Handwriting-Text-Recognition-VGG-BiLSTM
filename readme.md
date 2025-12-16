# Handwritten Text Recognition using (CNN) VGG and BiLSTM

This pipeline was adapted from [Naveen Reddy Marthala](https://github.com/naveenmarthala/Handwritten-word-recognition-OCR----IAM-dataset---CNN-and-BiRNN.git) repo with some modification.

The dataset we used was from [IAM Word Dataset](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database). The dataset can be downloaded from here [Dataset](https://drive.google.com/drive/folders/1byeJdl4ohWqSftkaG6kJVC9skI0AkT9w?usp=sharing)

## 1. install requirement
I trained the model with python 3.9.23 using tensorflow GPU. <br>
<i>Build environment using conda (optional)<i>
```python
conda create -n hwtr python==3.9.23
```
<br>
install the requrements.
```python
pip install -r requirement.txt
```

## 2. move to src folder and run demo
```python
cd src
python demo.py
```