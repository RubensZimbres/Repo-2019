# PDF - Tesseract + Text-To-Speech API  

```
$ sudo apt update
$ sudo apt install tesseract-ocr
$ sudo apt install libtesseract-dev
```  
```
$ sudo vi /etc/ImageMagick-6/policy.xml
```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Tesseract/Pics/tesseract.PNG>  

```
$ convert -density 300 /path/to/my/document.pdf -depth 8 -strip -background white -alpha off file.tiff

$ tesseract file.tiff output.txt
```  

