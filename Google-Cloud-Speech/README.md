# Google Cloud Speech

<b>FFMPEG</b>  

```
$ ffmpeg -i ggf_1.mp3 -ac 2 -ab 128k -af silenceremove=0:0:0:1:5:-25dB -filter:a volume=1.5:c0 -filter:a equalizer=f=8000:t=h:w=1:g=-5 -filter:a dynaudnorm ggf_output_other1.5-5.wav
```
<b>Where:</b>  

-silenceremove: start_periods:start_duration:start_threshold:stop_periods:stop_duration:stop_threshold
-ac – Set the number of audio channels (1,2)  
-ab – Indicates the audio bitrate (8k, 128k, 256k)  
-ar – Set the audio frequency of the output file (22050, 44100, 48000 Hz)  
-filter:a  
f – central frequency in Hz  
width_type – for defining the bandwidth, can be one of h (Hz), q (Q), o (octave) or s (slope)  
w – the value of the chosen bandwidth  
g – the gain  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-Cloud-Speech/Pics/audio_ggf0.png width="865" height="600">  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-Cloud-Speech/Pics/ffmpeg_analysis.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Google-Cloud-Speech/Pics/match1.png>  
