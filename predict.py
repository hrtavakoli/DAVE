#
# DAVE: A Deep Audio-Visual Embedding for Dynamic Saliency Prediction
# https://arxiv.org/abs/1905.10693
# https://hrtavakoli.github.io/DAVE/
#
# Copyright by Hamed Rezazadegan Tavakoli
#

import re
import os

import torch
import numpy as np

from PIL import Image
from utils.process_video_audio import LoadVideoAudio

from model import DAVE


# the folder find the videos consisting of video frames and the corredponding audio wav
VIDEO_TEST_FOLDER = './data/'
# where to save the predictions
OUTPUT = './result'
# where tofind the model weights
MODEL_PATH = './weights/model.pth.tar'

# some config parameters

IMG_WIDTH = 256
IMG_HIGHT = 320
TRG_WIDTH = 32
TRG_HIGHT = 40

device = torch.device("cuda:0")


class PredictSaliency(object):

    def __init__(self):
        super(PredictSaliency, self).__init__()

        self.video_list = [os.path.join(VIDEO_TEST_FOLDER, p) for p in os.listdir(VIDEO_TEST_FOLDER)]
        self.model = DAVE()
        self.model.load_state_dict(self._load_state_dict_(MODEL_PATH), strict=True)
        self.output = OUTPUT

        self.model = self.model.cuda()
        self.model.eval()

    @staticmethod
    def _load_state_dict_(filepath):
        if os.path.isfile(filepath):
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath, map_location=device)

            pattern = re.compile(r'module+\.*')
            state_dict = checkpoint['state_dict']
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = re.sub('module.', '', key)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        return state_dict

    def predict(self, stimuli_path, fps, out_path):

        if not os.path.exists(out_path):
            os.mkdir(out_path)

        video_loader = LoadVideoAudio(stimuli_path, fps)
        vit = iter(video_loader)
        for idx in range(len(video_loader)):
            video_data, audio_data = next(vit)
            video_data = video_data.cuda()
            audio_data = audio_data.cuda()
            video_data = torch.unsqueeze(video_data, 0)
            audio_data = torch.unsqueeze(audio_data, 0)
            prediction = self.model(video_data, audio_data)

            saliency = prediction.cpu().data.numpy()
            saliency = np.squeeze(saliency)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            saliency = Image.fromarray((saliency*255).astype(np.uint8))
            saliency = saliency.resize((640, 480), Image.ANTIALIAS)
            saliency.save('{}/{}.jpg'.format(out_path, idx+1), 'JPEG')

    def predict_sequences(self):

        for v in self.video_list:
            sample_rate = int(v[-2:])
            bname = os.path.basename(v[:-3])
            output_path = os.path.join(self.output, bname)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            self.predict(v, sample_rate, output_path)


if __name__ == '__main__':

    p = PredictSaliency()
    # predict all sequences
    p.predict_sequences()
    # alternatively one can call directy for one video
    #p.predict(VIDEO_TO_LOAD, FPS, SAVE_FOLDER) # the second argument is the video FPS.

