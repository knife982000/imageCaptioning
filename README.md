# Image Captioning
Image Camption is a neural network based on the [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Instead of using a LSTM as RNN, the implementation uses a GRU. The code is based on the Tensorflow implementation of a GRUCell and the [Tensorflow tutorial](https://www.tensorflow.org/tutorials/text/image_captioning#preprocess_and_tokenize_the_captions).

## Requirements
This code has been tested with Tensorflow 2.1.0 and 2.3.0

## Files:

* download.py: it downloads the training dataset. This step is required before training.
* preprocess_data.py: it preprocess the training dataset. It extract the image feautres using the keras pretrained InceptionV3 NN. This step is required before training.
* trainModel.py. it train a model for 10 epochs. It might be enough for the first training, but I would recommend to train it at least for 30 epochs. Further fine-tuning could be applied. It generates a tokenizer.pickle file, for tokenize the string captions, and a model file per each training epoch.
* showAndTell.py: it is not an executable, it defines the GRURecurrent layer.
* trainModel.py: it generates captions for a list of pictures using a given model using beam search. By defualt, the beam search is of width 3 and the maximum length of the generated captions is 20. 

```
python trainModel.py -m model.h5 -p pict1.jpg pict2.jpg ... pictN.jpg
```
## Pretrained model

A pre-trained model is abailable at [this link](https://mega.nz/file/M5Z0ADgR#468oecjFSxN1fc875vt5KLDqmzI8dNtecTPuW72A4Nc). 

