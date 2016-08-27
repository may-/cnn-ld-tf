# Convolutional Neural Network for Language Detection

**Note:** This is mostly based on https://github.com/yuhaozhang/sentence-convnet

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
    Access to http://localhost:5050/docs/
    
---


## Requirements

- [Python 2.7](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/) (tested with version 0.10.0rc0)
- [Numpy](http://www.numpy.org/)

To download TED corpus

- [Beautifulsoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pandas](http://pandas.pydata.org/)

Web API

- [Flask](http://flask.pocoo.org/)


## Data

+ Language detection  
    `./data/ted500` directory includes preprocessed data.
    If you want to download original data, please run the script:
    ```sh
    python ./ted.py
    ```
    You will need 2GB+ disk space.

+ Sentiment analysis  
    `./data/mr` directory includes 

+ Pretrained word embeddings  
    To use the pretrained word2vec embeddings, download the Google News 
    pretrained vector data from [this Google Drive link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), 
    and unzip it to the `./data/word2vec` directory. It will be a `.bin file.


**Attention:** Data file encoding must be utf-8.

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
 
Details: please visit [documentation](https://may-.github.io/cnn-ld-tf/stats.hrml)
    