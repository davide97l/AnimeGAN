# GAN

## How to use

Train a GAN model on the images contained in the directory "dataset" for 10000 epochs:
```
$ python DCGAN.py -n name_project -d dataset -t 10000
```
Load the saved model and continue training it for 10000 more epochs:
```
$ python DCGAN.py -n name_project -d dataset -t 20000 -i 10000
```
Generate some images with the traine model saved at the 10000 epoch:
```
$ python DCGAN.py -n name_project -i 10000 -m 1 -w 4
```
See more options:
```
$ python DCGAN.py -h
```
