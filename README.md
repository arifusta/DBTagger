# DBTagger: Multi-Task Learning for Keyword Mapping in NLIDBs Using Bi-Directional Recurrent Neural Networks

DBTagger is a keyword mapper, proposed to be used in Natural Language Interfaces in Databases. It outputs schema tags of tokens in the natural language query. It is an end-to-end and schema independent solution. It does not require any pre-processing for the input natural language query to determine schema tags such as tables, columns or values. This repository contains source code and annotated datasets used in the [paper](https://dl.acm.org/doi/10.14778/3446095.3446103), which will be published in PVLDB'21.

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
For training use [multiTask.py](multiTask.py) script. The script takes 6 parameters which are as follows:
- `schemaName:` name of the schema. In order to differentitate the actual datasets and schemas in Spider with the same schemaName (e.g. imdb), provide `imdb2` to refer the schema in Spider dataset.
- `dropout:` DBTagger utilizes RNNs with GRU units. For training, the model requires a dropout value to train the model.
- `epoch1:` Number of epochs for Adadelta optimizer
- `epoch2:` Number of epochs for Nadam optimizer
- `skip:` The flag to indicate whether to use cross-skip connections as explained in the paper. True or False
- `gpu_id:` The gpu_id to train the model on.

An example call usage is:
```
python multiTask.py imdb 0.5 100 200 skip 1
```

--------
## Evaluation on Pre-trained Models
[Models](https://github.com/arifusta/DBTagger/tree/main/Models) folder contains three pre-trained models for `imdb`, `scholar` and `yelp` datasets. To evaluate test sentences on those pre-trained models you can use [evalute.py](evaluate.py) script. It takes two parameters:
- `schemaName:` name of the schema.
- `gpu_id:` The gpu_id to run the model on.

Predictions will be in a `.txt` file inside the [Results/Evaluation](https://github.com/arifusta/DBTagger/tree/main/Results/Evaluation) folder.

An example usage is:
```
python evaluate.py imdb 0
```
--------
## Citing
If you find this repository helpful, feel free to cite our publication [DBTagger](https://dl.acm.org/doi/10.14778/3446095.3446103):
``` 
@article{10.14778/3446095.3446103,
    author = {Usta, Arif and Karakayali, Akifhan and Ulusoy, \"{O}zg\"{u}r},
    title = {DBTagger: Multi-Task Learning for Keyword Mapping in NLIDBs Using Bi-Directional Recurrent Neural Networks},
    year = {2021},
    issue_date = {January 2021},
    publisher = {VLDB Endowment},
    volume = {14},
    number = {5},
    issn = {2150-8097},
    url = {https://doi.org/10.14778/3446095.3446103},
    doi = {10.14778/3446095.3446103},
    journal = {Proc. VLDB Endow.},
    month = jan,
    pages = {813–821},
    numpages = {9},
}
```

