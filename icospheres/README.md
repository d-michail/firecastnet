
## Build icospheres

```
docker run -it --rm -v `pwd`:/models pymesh/pymesh /bin/bash
cd /models
./build_icospheres.py
CTRL-D
```

You will get a file called `icospheres.json`. It will be owned by root. Make sure to chown it to your user. 


