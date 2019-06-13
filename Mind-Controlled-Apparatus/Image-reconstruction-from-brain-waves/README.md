# Image Reconstruction from Brain Waves  

<b>Procedures:</b>  

Brain Wave reader: Muse: 4 points: 2 in frontal, 2 in temporal.

```
$ chmod +x museresearchtools-3.4.2-linux-installer.run
$ ./museresearchtools-3.4.2-linux-installer.run
$ sudo apt-get install libxrender1:i386 libxtst6:i386 libxi6:i386
$ sudo apt-get -f install
$ ./MuseLab
```

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Mind-Controlled-Apparatus/Pics/Muse_Monitor/Screenshot_20190610-164617_Muse%20Monitor.jpg width="250" height="400"> <img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Mind-Controlled-Apparatus/Pics/Muse_Monitor/Screenshot_20190610-164542_Muse%20Monitor.jpg width="250" height="400"> <img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Mind-Controlled-Apparatus/Pics/Muse_Monitor/Screenshot_20190610-164515_Muse%20Monitor.jpg width="250" height="400"> <img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Mind-Controlled-Apparatus/Pics/Muse_Monitor/Screenshot_20190610-164341_Muse%20Monitor.jpg width="250" height="400"> <img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Mind-Controlled-Apparatus/Pics/Muse_Monitor/Screenshot_20190610-161054_Muse%20Monitor.jpg width="250" height="400">   

<b>Procedures:</b>  

1. See a picture of a dog and save brain wave data from Alpha, Beta, Gamma and Theta waves (variable muse in notebook - museMonitor_train1.csv)  

2. Develop a convolutional autoencoder with brain wave data as input and dog image as output  

3. Train the autoencoder and save weights  

4. Predict image from new brain waves (variable muse2 in notebook - museMonitor_test.csv)    

<b>Notebook:</b>  

<a href="https://github.com/RubensZimbres/Repo-2019/blob/master/Mind-Controlled-Apparatus/Image-reconstruction-from-brain-waves/Muse_start_reconstruct.ipynb" target="_blank">Jupyter Notebook</a>

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Mind-Controlled-Apparatus/Image-reconstruction-from-brain-waves/muse_dog.jpg width="380" height="380">       <img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Mind-Controlled-Apparatus/Image-reconstruction-from-brain-waves/reconstruct.png width="400" height="400">
