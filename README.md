# DBTagger: Multi-Task Learning for Keyword Mapping inNLIDBs Using Bi-Directional Recurrent Neural Networks

DBTagger is a keyword mapper, proposed to be used in Natural Language Interfaces in Databases. It outputs schema tags of tokens in the natural language query. It is an end-to-end and schema independent solution. It does not require any pre-processing for the input natural language query to determine schema tags such as tables, columns or values. This repository contains source code and annotated datasets; as used in the official paper, which will be published in PVLDB'21.

--------

## Software dependencies
Code is implemented in Python 3.6.8 using Keras 2.2.4 and Tensorflow-gpu 1.12.0 libraries. Other required libraries are listed in `spec-file.txt` as well. We recommend using a python distribution such as anaconda to install libraries.

```
conda create --name myenv --file spec-file.txt
```
--------
## Datasets
In our experiments we used publically available yelp, imdb ([SQLizer](https://dl.acm.org/doi/10.1145/3133887)), and mas ([NALIR](https://dl.acm.org/doi/10.14778/2735461.2735468)) datasets. In addition to these releational database dumps, we also used different schemas from the [Spider](https://yale-lily.github.io/spider) dataset; which are academic, college, hr, imdb, and yelp. 

The structure of dataset folders differs. For each non-Spider datasets, there are 6 .txt files; 3 for training and 3 for test purposes. Although, the datasets are divided into splits, the experiments in the paper are done using Cross-fold validation with merged data. Therefore each non-Spider dataset has following `.txt` files. These `.txt` files are inside [FixedDataset](https://github.com/arifusta/DBTagger/tree/main/FixedDataset). For each schema in Spider dataset, there is a distinct folder named as schemaName inside [Spider](https://github.com/arifusta/DBTagger/tree/main/FixedDataset/Spider) folder. Also note that original natural language questions are also present in the following tag files. Each natural language query in the following files are in the form of list of tokens seperated by space where tokens are given as `<orgWord>/<Tag>/`.
    
- `<schemaName><Train/Test><Pos>.txt:` Pos tags of the tokens for natural language queries in the dataset. Pos tags are output using Standford Pos Tagger. 
- `<schemaName><Train/Test><Tag>.txt:` Type tags of the tokens for natural language queries in the dataset. These tags are annotated manually.
- `<schemaName><Train/Test><DbTag>.txt:` Schema tags of the tokens for natural language queries in the dataset. These tags are annotated manually

--------
## Training
For training use [Train_POS.py](Train_POS.py) script. The script takes 6 parameters which are as follows:
-
