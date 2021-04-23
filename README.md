# Dolly Zoom


## setup
Several functions are implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided [binary packages](https://docs.cupy.dev/en/stable/install.html#installing-cupy) as outlined in the CuPy repository. Please also make sure to have the `CUDA_HOME` environment variable configured.

In order to generate the video results, please also make sure to have `pip install moviepy` installed.

## usage
To run it on a video and generate the Vertigo effect (Dolly Zoom) fully automatically, use the following command.

first edit the following three lines of dollyzoom.py
```
    arguments_strIn = ['./images/input.mp4']
    arguments_strOut = './output'
    starter_zoom = 2
```
Then run
```
python dollyzoom.py'
```
