#!/usr/bin/env python

import torch
import cv2
import getopt
import math
import moviepy.editor
import numpy
import sys
import matplotlib.pyplot as plt


def showImage(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


##########################################################

assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 12)  # requires at least pytorch version 1.2.0
torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

from common import process_load
from common import process_autozoom
from common import process_kenburns

##########################################################
for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--in' and strArgument != '': arguments_strIn = strArgument  # path to the input image
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored
# end

##########################################################
if __name__ == '__main__':
    arguments_strIn = './images/input.mp4'
    arguments_strOut = './results/output'
    starter_zoom = 2

    clip_counter = 1
    clip_frame = 50
    zoom_step = (starter_zoom-1)/clip_frame

    npylistResult = []
    cap = cv2.VideoCapture(arguments_strIn)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        img = frame
        if frame is None or clip_counter>3:
            cap.release()
            moviepy.editor.ImageSequenceClip(
                sequence=[npyFrame[:, :, ::-1] for npyFrame in list(reversed(npylistResult))[1:] + npylistResult],
                fps=25).write_videofile(arguments_strOut + '_' + str(clip_counter) + '.mp4')
            break

        if frame_count>=clip_frame:
            moviepy.editor.ImageSequenceClip(
                sequence=[npyFrame[:, :, ::-1] for npyFrame in list(reversed(npylistResult))[1:] + npylistResult],
                fps=25).write_videofile(arguments_strOut+'_'+str(clip_counter)+'.mp4')
            clip_counter = clip_counter+1
            npylistResult = []
            frame_count = 0

        fltZoom = starter_zoom - zoom_step*frame_count

        print('frame_count:', frame_count, img.shape, fltZoom)
        # showImage(img)

        npyImage = img
        intWidth = npyImage.shape[1]
        intHeight = npyImage.shape[0]

        fltRatio = float(intWidth) / float(intHeight)

        intWidth = min(int(1024 * fltRatio), 1024)
        intHeight = min(int(1024 / fltRatio), 1024)

        npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

        process_load(npyImage, {})

        objFrom = {
            'fltCenterU': intWidth / 2.0,
            'fltCenterV': intHeight / 2.0,
            'intCropWidth': int(math.floor(intWidth)),
            'intCropHeight': int(math.floor(intHeight))
        }

        if frame_count == 0:
            objTo_First = process_autozoom({
                'fltShift': 0,
                'fltZoom': fltZoom,
                'objFrom': objFrom
            })
            objTo = objTo_First.copy()
        else:
            objTo = {
                'fltCenterU': objTo_First['fltCenterU'],
                'fltCenterV': objTo_First['fltCenterV'],
                'intCropWidth': int(round(objFrom['intCropWidth'] / fltZoom)),
                'intCropHeight': int(round(objFrom['intCropHeight'] / fltZoom))
            }

        npyResult = process_kenburns({
            'fltSteps': numpy.linspace(0.0, 1.0, 2).tolist(),
            'objFrom': objFrom,
            'objTo': objTo,
            'process_kenburns': True,
            'fltZoom': fltZoom,
            'fltStartZoom': starter_zoom
        })

        npylistResult.append(npyResult[1])

        frame_count = frame_count + 1


