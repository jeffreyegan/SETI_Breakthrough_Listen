# SETI_Breakthrough_Listen
An entry in a Kaggle AI/ML competition to find extraterrestrial technosignatures in spectrogram data from deep space observations using radio telescopes.



# Running the models and notebooks within Docker
An explanation of some non-standard arguments and shared volumes:
- Ports for Jupyter Notebook
- GPU Passthrough for CUDA-accelerated code
- Mount shm to increase shared memory, prevent out of memory efforts on large batch sizes
- Mount the poject repository
- Mount the data store

For the interactive situation, the run command is augmented with an interactive flag and an entrypoint to preempt the automated execution.


### Automated - Train and Test Model
Running the docker container with the following command will cause the container to automatically train, save, and test the model detailed in `seti_bl_pytorch_cnn.py`.
```
docker run -p 8888:8888 --gpus all --rm -v /dev/shm:/dev/shm -v /home/jeffrey/repos/SETI_Breakthrough_Listen:/SETI -v /home/jeffrey/data/seti_bl:/data  setibl:0.1.0
```

### Interative & Jupyter Notebook
To run the container and either work within the environment interactively or run the jupyter notebook, use the following command:

```
docker run -p 8888:8888 --gpus all --rm -v /dev/shm:/dev/shm -v /home/jeffrey/repos/SETI_Breakthrough_Listen:/SETI -v /home/jeffrey/data/seti_bl:/data -it --entrypoint bash setibl:0.1.0
```
And then at the container's bash prompt, use the following command to run the notebooks:

```
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```
Any other python or tasks requireing the environment may be performed here as well.
