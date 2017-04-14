# Convolutional Neural Network for Language Detection

**Note:** This project is mostly based on https://github.com/yuhaozhang/sentence-convnet

---

## Demo

1. Run API Server

    ```sh
    python ./main.py
    ```

2. Run HTML server  
    for example:
    ```
    python -m SimpleHTTPServer 5050
    ```
    Access to [http://localhost:5050/docs/](http://localhost:5050/docs/)
    
---


## Requirements

- [Python 2.7](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/) (tested with version <strike>0.10.0rc0</strike> -> 1.0)
- [Numpy](http://www.numpy.org/)

To train with pretrained embedding (`train.py --use_pretrain=True`)

- [Gensim](https://radimrehurek.com/gensim/)

To download TED corpus (`ted.py`)

- [Beautifulsoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pandas](http://pandas.pydata.org/)

To visualize (`visualize.ipynb`)

- [Scikit-learn](http://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)

Web API (`main.py`)

- [Flask](http://flask.pocoo.org/)


## Data
+ TED Subtitle Corpus  
    `./data/ted500` directory includes preprocessed data.
    To reproduce (2GB+ disk space required):
    ```sh
    python ./ted.py
    ```
    
+ Your own data  
    Put the data file per class, e.g. `class_names = ['neg', 'pos']`:
    ```
    cnn-ld-tf
    ├── ...
    └── data
        └── mr
            ├── mr.neg  # examples with class neg
            └── mr.pos  # examples with class pos
    ```    

    **Note:**
    
    + Data file encoding must be utf-8.
    + One example per line.
    + The number of examples of each class must be the same.

## Preprocess

```sh
python ./util.py
```

## Training

```sh
python ./train.py
```

## Prediction

```sh
python ./predict.py
```

## Evaluation

```sh
python ./eval.py
```

## Run TensorBoard

```sh
tensorboard --logdir=./model/ted500/summaries
```
![Embeddings by script name](./docs/img/tensorboard.png)


## References

CNN for text classification:

* https://github.com/yuhaozhang/sentence-convnet
* https://github.com/dennybritz/cnn-text-classification-tf
* http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
* http://tkengo.github.io/blog/2016/03/14/text-classification-by-cnn/

TED Corpus:

* https://github.com/ajinkyakulkarni14/How-I-Extracted-TED-talks-for-parallel-Corpus-

Language Detection:

* http://drops.dagstuhl.de/opus/volltexte/2014/4574/pdf/22.pdf

Web API on heroku:

* https://github.com/sfujiwara/pn2bs


## Pre-trained model
     
* Supported languages (65):  
   `["ar", "az", "bg", "bn", "bo", "cs", "da", "de", "el", "en", "es",
     "fa", "fi", "fil", "fr", "gu", "he", "hi", "ht", "hu", "hy", "id",
     "is", "it", "ja", "ka", "km", "kn", "ko", "ku", "lt", "mg", "ml",
     "mn", "ms", "my", "nb", "ne", "nl", "nn", "pl", "ps", "pt", "ro",
     "ru", "si", "sk", "sl", "so", "sq", "sv", "sw", "ta", "te", "tg",
     "th", "tl", "tr", "ug", "uk", "ur", "uz", "vi", "zh-cn", "zh-tw"]` 
 
Details: please visit [documentation](https://may-.github.io/cnn-ld-tf/stat.html)
    
