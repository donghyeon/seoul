# Seoul Urban Data Science Research

## Download PM dataset
Airkorea PM dataset:
[2014.zip](http://www.airkorea.or.kr/file/download/?atch_id=21627)
[2015.zip](http://www.airkorea.or.kr/file/download/?atch_id=22060)
[2016.zip](http://www.airkorea.or.kr/file/download/?atch_id=48522)
[2017.zip](http://www.airkorea.or.kr/file/download/?atch_id=71690)

## Requirements
[TensorFlow Official Models](https://github.com/tensorflow/models/tree/master/official)

## RNN based models
Download and unzip files. You can find a few auxiliary functions to manage dataset in
[read\_and\_preprocess\_pm.py](https://github.com/donghyeon/seoul/blob/master/read_and_preprocess_pm.py),
or just read pickled data [df\_pm.pickle](https://goo.gl/QYfZDH)
to reduce loading times.

Just run [seoul\_main.py](https://github.com/donghyeon/seoul/blob/master/seoul_main.py).
You can change the model function of the estimator.