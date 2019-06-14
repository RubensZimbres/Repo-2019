# OpenAI GPT-2 

```
$ git clone https://github.com/openai/gpt-2.git && cd gpt-2  

$ pip3 install tensorflow-gpu==1.12.0  

$ pip3 install -r requirements.txt  

$ python3 download_model.py 345M  

$ docker build --tag gpt-2 -f Dockerfile.gpu . # or Dockerfile.cpu  

$ docker run --runtime=nvidia -it gpt-2 bash  

$ export PYTHONIOENCODING=UTF-8  

# Unconditional sample generation  

$ python3 src/generate_unconditional_samples.py --top_k 40 --temperature 0.7 | tee /tmp/samples

# Conditional sample generation

$ python3 src/interactive_conditional_samples.py --temperature 0.7 --top_k 5 --model_name 345M --seed 149 --nsamples 80 
```

```
$ gcloud auth login

$ gsutil cp /home/gpt-2/* gs://gpt-2-storage
```

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/OpenAI-GPT-2/PIcs/gpt-2_1.JPG>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/OpenAI-GPT-2/PIcs/gpt-2_4.JPG>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/OpenAI-GPT-2/PIcs/gpt-2_4.1.JPG>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/OpenAI-GPT-2/PIcs/gpt-2_5.JPG>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/OpenAI-GPT-2/PIcs/gpt-2-02.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/OpenAI-GPT-2/PIcs/gpt-2-03.png>  

