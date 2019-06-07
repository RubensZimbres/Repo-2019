# Google's EfficientNets  

Paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks  

https://arxiv.org/pdf/1905.11946.pdf

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-EfficientNet/Pics/efficient.png>  

<b>Using Checkpoints</b>

```
$ git clone https://github.com/tensorflow/tpu/

# CIFAR-10
$ wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
$ wget https://raw.githubusercontent.com/tiagosn/cifar-10_py2png/master/cifar-10_py2png.py
$ python2.7 cifar-10_py2png.py cifar-10-batches-py

#IMAGENET
$ wget https://raw.githubusercontent.com/RubensZimbres/Repo-2019/master/Google-EfficientNet/download_imagenet.sh
$ mkdir data && cd data && sudo bash download_imagenet.sh

$ export MODEL=efficientnet-b3
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-b3.tar.gz
$ mkdir weights
$ tar -xvf efficientnet-b3.tar.gz
$ gsutil cp home/rubens/weights/* gs://efficient-net
$ wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt
$ python eval_ckpt_main.py --model_name=efficientnet-b3 --ckpt_dir=efficientnet-b3 --example_img=panda.jpg --labels_map_file=labels_map.txt
```

<b>Training</b>  

```
$ export PYTHONPATH="$PYTHONPATH:/home/rubensvectomobile"
$ python main.py --tpu=rubens --data_dir=gs://efficient-net/data --model_dir=gs://efficient-net

$ python main.py --use_tpu=False --data_dir=gs://efficient-net/data --model_dir=gs://efficient-net
```
