https://linuxconfig.org/how-to-change-default-python-version-on-debian-9-stretch-linux

```
$ python --version

$ ls /usr/bin/python*

$ update-alternatives --install /usr/bin/python python /usr/bin/python3.7.4 1

$ update-alternatives --config python
```
