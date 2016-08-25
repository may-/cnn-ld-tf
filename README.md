# Convolutional Neural Network for Language Detection

**Note:**  
This is mostly based on https://github.com/yuhaozhang/sentence-convnet

---


## Requirements

- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)

To download TED corpus

- [Beautifulsoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pandas](http://pandas.pydata.org/)

Web API

- [Flask](http://flask.pocoo.org/)

## Preprocess

Download TED Corpus
```sh
python ./ted.py
```

Prepare train/test Data  
```sh
python ./util.py
```

## Training

```sh
python ./train.py
```

## Prediction

```python
import predict

result = predict.predict(u'日本語のテスト')
print result['prediction']
```

## Evaluation

```sh
python ./eval.py --train_dir=./model/ted500
```

## Run TensorBoard

```sh
tensorboard --logdir=./model/ted500/summaries
```

## Use as Web API

### Run API Server

```sh
python ./main.py
```

### Request

```
http://localhost:5000/predict?text="日本語のテスト"
```

### Response

```
{
  "prediction": "ja", 
  "scores": {
    "en": -1344.27,
    "fr": -1003.39,  
    "de": -788.429, 
    "ja": 1795.52,
        .
        .
        .
  }
}
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

* Corpus domain: speech transcriptions
* Number of train examples: 29250 (450 per language)
* Number of dev examples: 3250 (50 per language)
* Parameters:
    + `data_dir = './data/ted500'`
    + `train_dir = './model/ted500'`
    + `batch_size = 100`
    + `emb_size = 300`
    + `num_kernel = 100`
    + `min_window = 3`
    + `max_window = 5`
    + `vocab_size = 4090`
    + `num_classes = 65`
    + `sent_len = 259`
    + `l2_reg = 0.0`
    + `optimizer = 'adam'`
    + `init_lr = 0.01`
    + `lr_decay = 0.95`
    + `tolerance_step = 500`
    + `dropout = 0.5`
    + `log_step = 10`
    + `summary_step = 200`
    + `save_epoch = 5`
     
* Supported languages (65):
   `["ar", "az", "bg", "bn", "bo", "cs", "da", "de", "el", "en", "es",
     "fa", "fi", "fil", "fr", "gu", "he", "hi", "ht", "hu", "hy", "id",
     "is", "it", "ja", "ka", "km", "kn", "ko", "ku", "lt", "mg", "ml",
     "mn", "ms", "my", "nb", "ne", "nl", "nn", "pl", "ps", "pt", "ro",
     "ru", "si", "sk", "sl", "so", "sq", "sv", "sw", "ta", "te", "tg",
     "th", "tl", "tr", "ug", "uk", "ur", "uz", "vi", "zh-cn", "zh-tw"]` 
    