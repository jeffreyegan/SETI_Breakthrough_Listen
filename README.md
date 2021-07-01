# SETI_Breakthrough_Listen
An entry in a Kaggle AI/ML competition to find extraterrestrial signals in data from deep space.


docker run -p 8888:8888 --gpus all --rm -v /dev/shm:/dev/shm -v /home/jeffrey/repos/SETI_Breakthrough_Listen/notebooks:/notebooks -v /home/jeffrey/data/seti_bl:/data -it setibl:0.0.27