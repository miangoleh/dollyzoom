# Dolly Zoom


## setup
Test with Python 3.6 and Pytorch 1.6. 

Several functions are implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided [binary packages](https://docs.cupy.dev/en/stable/install.html#installing-cupy) as outlined in the CuPy repository. Please also make sure to have the `CUDA_HOME` environment variable configured.

In order to generate the video results, please also make sure to have `pip install moviepy` installed.

Three different Repositories are used in this repo.

Download Midas model weights from https://github.com/intel-isl/MiDaS. Put the weight in the following path : 
```
midas/model-f46da743.pt
```
Download depthmerge model weights from https://github.com/ouranonymouscvpr/cvpr2021_ouranonymouscvpr. Put the weights in the following path :
```
depthmerge/checkpoints/scaled_04_1024/latest_net_G.pth
```

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

## Acknoledgment
We borrowed some parts of the the following papers and their implementation for our project

Midas https://github.com/intel-isl/MiDaS
```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

3D ken Burns effect from a single image https://github.com/sniklaus/3d-ken-burns
```
@article{Niklaus_TOG_2019,
         author = {Simon Niklaus and Long Mai and Jimei Yang and Feng Liu},
         title = {3D Ken Burns Effect from a Single Image},
         journal = {ACM Transactions on Graphics},
         volume = {38},
         number = {6},
         pages = {184:1--184:15},
         year = {2019}
     }
     
Pix2Pix https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```



