# SETI_Breakthrough_Listen
An entry in a Kaggle AI/ML competition to find extraterrestrial signals in data from deep space.



# Docker Command
An explanation of some non-standard arguments and shared volumes:
- Ports for Jupyter Notebook
- GPU Passthrough for CUDA-accelerated code
- Mount shm to increase shared memory, prevent out of memory efforts on large batch sizes
- Mount the poject repository
- Mount the data store

```
docker run -p 8888:8888 --gpus all --rm -v /dev/shm:/dev/shm -v /home/jeffrey/repos/SETI_Breakthrough_Listen:/SETI -v /home/jeffrey/data/seti_bl:/data -it setibl:0.0.27

docker run -p 8888:8888 --rm -v /dev/shm:/dev/shm -v /home/jeffrey/repos/SETI_Breakthrough_Listen:/SETI -v /home/jeffrey/data/seti_bl:/data setibl:0.0.27
```

### Interative

jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
python seti_bl_pytorch_cnn.py