import requests

REST_API_URL = "https://location-project02-12345.cloudfunctions.net/test-function"

from google.cloud import storage

client = storage.Client()

bucket = client.get_bucket('images22')

blob = bucket.get_blob('dog.jpg')

data = blob.download_to_filename('dog22.jpg')

with open("dog22.jpg", 'rb') as f:
    image=f.read()

payload = {"image": image}

r = requests.post(REST_API_URL, files=payload).json()

print(r)
