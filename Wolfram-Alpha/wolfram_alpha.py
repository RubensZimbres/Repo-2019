import wolframalpha
client = wolframalpha.Client("AB1234-12345ABCD")

res = client.query('temperature in Los Angeles, CA on October 12, 2000')

next(res.results).text

for pod in res.pods:
    for sub in pod.subpods:
        print(sub.text)
