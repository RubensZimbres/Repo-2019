# Airflow

```
$ export SLUGIFY_USES_TEXT_UNIDECODE=yes
$ export LC_ALL='en_US.UTF-8'
$ export LC_CTYPE='en_US.UTF-8'
$ sudo apt install libpq-dev python3-dev
$ pip3 install psycopg2
$ airflow initdb
$ airflow version
```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Airflow/Pics/ariflow0.png>  

```
$ vi airflow.cfg
 
executor = LocalExecutor
sql_alchemy_conn = postgresql+psycopg2://user:password@host:port/airflow
load_examples = False

$ airflow initdb
```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Airflow/Pics/airflow1.png>  

Create first_dag.py and second_dag.py inside /airflow/dags/ and 

```
$ airflow scheduler -D
$ pip install Flask==1.1.1
$ airflow webserver -p 8888
```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Airflow/Pics/airflow22.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Airflow/Pics/airflow44.png>  
