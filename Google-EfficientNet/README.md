# Google's EfficientNets  

Paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks  

https://arxiv.org/pdf/1905.11946.pdf

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-EfficientNet/Pics/efficient.png>  

# Using Checkpoints

<b> ATTACH AND CONFIGURE GOOGLE COMPUTE ENGINE DISK </b>

```
$ sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
$ sudo mkdir -p /mnt/disks/
$ sudo mount -o discard,defaults /dev/sdb /mnt/disks/
$ sudo chmod a+w /mnt/disks/
$ sudo cp /etc/fstab /etc/fstab.backup
$ sudo blkid /dev/sdb
$ sudo vi /etc/fstab

UUID=3f228fd9-1dce-4197-b1b2b3-12345 /mnt/disks/ ext4 discard,defaults,nobootwait 0 2
# ESC : wq

$ sudo lsblk

NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
sda      8:0    0   60G  0 disk 
└─sda1   8:1    0   60G  0 part /
sdb      8:16   0  500G  0 disk /mnt/disks
$ cd /mnt/disks
```  

<b>Download IMAGENET --  SIZE = 155 GB</b>

```
$ pip install kaggle
$ export KAGGLE_USERNAME=rubens
$ export KAGGLE_KEY=xxxxxxxxxxxxxx
$ touch kaggle.json
$ vi kaggle.json # ADD CREDENTIALS
/mnt/disks/$ ~/.local/bin/kaggle competitions download -c imagenet-object-localization-challenge
/mnt/disks/$ tar -xvf imagenet_object_localization.tar.gz
$ du -hs

# OR

$ mkdir data && cd data
$ wget https://raw.githubusercontent.com/RubensZimbres/Repo-2019/master/Google-EfficientNet/download_imagenet.sh
$ sudo bash download_imagenet.sh

$ exit
$ sudo chmod -R +X /home
$ ...

ILSVRC/Annotations/CLS-LOC/train/ILSVRC2012_train_0012345.JPEG
ILSVRC/Data/CLS-LOC/train/ILSVRC2012_train_0012345.JPEG
ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_0012345.JPEG
ILSVRC/ImageSets/CLS-LOC/train_loc.txt
ILSVRC/ImageSets/CLS-LOC/train_cls.txt
ILSVRC/ImageSets/CLS-LOC/val.txt
ILSVRC/ImageSets/CLS-LOC/test.txt

$ cd ILSVRC/Data/CLS-LOC/train/n04562935/
$ gcloud auth login
$ sudo chmod -R 755 /mnt/disks/
$ cp $(ls | head -n 1000) /path/
$ gsutil cp /mnt/disks/ILSVRC/Data/CLS-LOC/train/n04562935/* gs://efficient-net/data/train
$ gsutil cp /mnt/disks/ILSVRC/Annotations/CLS-LOC/train/n04562935/* gs://efficient-net/data/train
$ gsutil du -s gs://efficient-net
```  

<img src=https://raw.githubusercontent.com/RubensZimbres/Repo-2019/master/Google-EfficientNet/Pics/kaggle_download1.png>  

<b>Project</b>

```
$ export MODEL=efficientnet-b0
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-b0.tar.gz
$ mkdir weights
$ tar -xvf efficientnet-b0.tar.gz
$ gsutil cp home/rubens/weights/* gs://efficient-net
$ wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt
$ python eval_ckpt_main.py --model_name=efficientnet-b3 --ckpt_dir=efficientnet-b3 --example_img=panda.jpg --labels_map_file=labels_map.txt
```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-EfficientNet/Pics/panda.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-EfficientNet/Pics/efficient0.png>  

# Training  

```
$ git clone https://github.com/tensorflow/tpu

$ mkdir train
  $ cd train
  $ tar xvf ~/Downloads/ILSVRC2012_img_train.tar
  $ find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  ### extract val data
  $ cd ..
  $ mkdir val
  $ cd val
  $ tar xvf ~/Downloads/ILSVRC2012_img_val.tar
  $ wget
https://raw.githubusercontent.com/jkjung-avt/jkjung-avt.github.io/master/assets/2017-12-01-ilsvrc2012-in-digits/valprep.sh
  $ bash ./valprep.sh

$ find train/ -name "*.JPEG" | wc -l

$ export PYTHONPATH="$PYTHONPATH:/home/rubens/efficient"
$ cd /tpu/models/official/efficientnet
$ python main.py --tpu=rubens --data_dir=gs://efficient-net/data --model_dir=gs://efficient-net
```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-EfficientNet/Pics/efficient_01.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-EfficientNet/Pics/efficient_00.png>  

