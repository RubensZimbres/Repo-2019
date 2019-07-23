# PDF - Tesseract + Text-To-Speech API  

```
$ sudo apt update
$ sudo apt install tesseract-ocr
$ sudo apt install libtesseract-dev
```  
```
$ sudo vi /etc/ImageMagick-6/policy.xml

$ identify -list resource
```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Tesseract/Pics/tesseract.PNG>  

```
$ convert -density 300 /home/rubens/document.pdf -depth 8 -strip -background white -alpha off file.tiff

$ convert -density 300 deeplearningbook.pdf -depth 8 -strip -background white -alpha off -resize 600x280 file4_deep.png

$ tesseract file.tiff output.txt
```  

