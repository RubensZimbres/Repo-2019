# Google's EfficientNets  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-EfficientNet/Pics/efficientnets.png>  

<b>Using Checkpoints</b>

```
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-b3.tar.gz
$ export MODEL=efficientnet-b3
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-b3.tar.gz
$ mkdir weights
$ cd weights && tar -xvf efficientnet-b3.tar.gz
$ gsutil cp home/rubens/weights/* gs://efficient-net
$ wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt
$ python eval_ckpt_main.py --model_name=efficientnet-b3 --ckpt_dir=efficientnet-b3 --example_img=panda.jpg --labels_map_file=labels_map.txt
```

<b>Training</b>  

```
$ export PYTHONPATH="$PYTHONPATH:/home/rubensvectomobile"
$ python main.py --tpu=rubens --data_dir=gs://efficient-net/data --model_dir=gs://efficient-net
```
