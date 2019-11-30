
# generic must imports
import os
import torch
import numpy as np


import utils.audio_params as audio_params
import librosa as sf
from utils.audio_features import waveform_to_feature


from PIL import Image
import torchvision.transforms.functional as F


__all__ = ['LoadVideoAudio']

#defined params @TODO move them to a parameter config file
DEPTH = 16
GT_WIDTH = 32
GT_HIGHT = 40

MEAN = [ 110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0 ]
STD = [ 38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0 ]

def adjust_len(a, b):
    # adjusts the len of two sorted lists
    al = len(a)
    bl = len(b)
    if al > bl:
        start = (al - bl) // 2
        end = bl + start
        a = a[start:end]
    if bl > al:
        a, b = adjust_len(b, a)
    return a, b


def create_data_packet(in_data, frame_number):

    n_frame = in_data.shape[0]

    frame_number = min(frame_number, n_frame) #if the frame number is larger, we just use the last sound one heard about
    starting_frame = frame_number - DEPTH + 1
    starting_frame = max(0, starting_frame) #ensure we do not have any negative frames
    data_pack = in_data[starting_frame:frame_number+1, :, :]
    n_pack = data_pack.shape[0]

    if n_pack < DEPTH:
        nsh = DEPTH - n_pack
        data_pack = np.concatenate((np.tile(data_pack[0,:,:], (nsh, 1, 1)), data_pack), axis=0)

    assert data_pack.shape[0] == DEPTH

    data_pack = np.tile(data_pack, (3, 1, 1, 1))

    return data_pack, frame_number


def load_wavfile(wav_file):
    """load a wave file and retirieve the buffer ending to a given frame

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

      frame_number: Is the frame to be extracted as the final frame in the buffer

    Returns:
      See waveform_to_feature.
    """
    wav_data, sr = sf.load(wav_file, sr=audio_params.SAMPLE_RATE, dtype='float32')
    assert sf.get_duration(wav_data, sr) > 1

    features = waveform_to_feature(wav_data, sr)

    return features


def get_wavFeature(features, frame_number):

    audio_data, valid_frame_number = create_data_packet(features, frame_number)
    return torch.from_numpy(audio_data).float(), valid_frame_number


def load_maps(file_path):
    '''
        Load the gt maps
    :param file_path: path the the map
    :return: a numpy array as floating number
    '''

    with open(file_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('L').resize((GT_HIGHT, GT_WIDTH), resample=Image.BICUBIC)
            data = F.to_tensor(img)
    return data


def load_video_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    assert int(frame_name[1:-4]) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - DEPTH+1
    start_frame_number = max(1, start_frame_number)
    frame_list = [f for f in range(start_frame_number, frame_number+1)]
    if len(frame_list) < DEPTH:
        nsh = DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)
    frames = []
    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:07d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
                img = F.to_tensor(img)
                img = F.normalize(img, MEAN, STD)
                frames.append(img)
    data = torch.stack(frames, dim=0)
    return data.permute([1, 0, 2, 3])


class LoadVideoAudio(object):
    """
        load the audio video
    """

    def __init__(self, stimuli_in, vfps):
        """
        :param stimuli_in:
        :param gt_in:
        """

        self.root_folder = stimuli_in
        self.sample = []
        fr = vfps

        video_frames = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]

        audio_file = [os.path.join(self.root_folder,  f) for f in os.listdir(self.root_folder)
                      if f.endswith('.wav')]

        self.audio_data = load_wavfile(audio_file[0])

        video_frames.sort()

        cnt = 0
        total_frame = str(len(video_frames))
        for video_frame in video_frames:
            frame_number = os.path.basename(video_frame)[0:-4]
            sample = {'total_frame': total_frame, 'fps': fr,
                      'frame': video_frame, 'frame_number': frame_number}
            self.sample.append(sample)
            cnt = cnt + 1

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, item):

        sample = self.sample[item]
        audio_params.EXAMPLE_HOP_SECONDS = 1/int(sample['fps'])
        audio_data, valid_frame_number = get_wavFeature(self.audio_data, int(sample['frame_number']))
        video_data = load_video_frames(sample['frame'], int(sample['frame_number']), valid_frame_number)

        return video_data, audio_data



if __name__ == "__main__":
   a = LoadVideoAudio('/ssd/VDA/test/clip_9_25', '/ssd/rtavah1/VIDEO_Saliency_database/annotation/maps/clip_9', 25)
   video_data, audio_data, gt_map = a.__getitem__(a.__len__()-1)
   print(a.__len__())
   print(video_data.shape)
   print(audio_data.shape)
   print(gt_map.shape)
