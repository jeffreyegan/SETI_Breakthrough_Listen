# SETI_Breakthrough_Listen
An entry in a Kaggle AI/ML competition to find extraterrestrial signals in data from deep space.


docker run --gpus all --rm -it deepdsp:0.0.1 -v ./project:./project

docker run -p 8888:8888 --gpus all -m=12g --rm -v /media/jeffrey/1947f403-ddfe-4e45-ae59-f2e6e395b2e9/data/seti_bl:/project/data -it setibl:0.0.10