# Seoul Urban Data Science Research

## Download PM dataset
Airkorea PM dataset:
[2014.zip](http://www.airkorea.or.kr/file/download/?atch_id=21627)
[2015.zip](http://www.airkorea.or.kr/file/download/?atch_id=22060)
[2016.zip](http://www.airkorea.or.kr/file/download/?atch_id=48522)
[2017.zip](http://www.airkorea.or.kr/file/download/?atch_id=71690)

Download and unzip files. You can find a few auxiliary functions to manage dataset in
[read\_and\_preprocess\_pm.py](https://github.com/donghyeon/seoul/blob/master/read_and_preprocess_pm.py),
or just read pickled data[df\_pm.pickle](https://goo.gl/QYfZDH)
to reduce loading times.

A simple LSTM experiments are added.
Just run [seoul\_lstm.py](https://github.com/donghyeon/seoul/blob/master/seoul_lstm.py)
