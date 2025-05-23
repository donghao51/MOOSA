from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

source_1 = [0, 1, 2, 5, 7]
source_2 = [0, 1, 4, 6, 7]
source_all = [0, 1, 2, 4, 5, 6, 7]
target_all = [0, 1, 3, 5, 6]

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', domain_name='source', domain=['D1'], modality='rgb', cfg=None, sample_dur=10, datapath='/path/to/EPIC-KITCHENS/'):
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.interval = 9
        self.sample_dur = sample_dur

        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            self.pipeline = Compose(train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.pipeline = Compose(val_pipeline)

        data1 = []

        if domain_name == 'source':
            source_dom1 = domain[0]
            train_file = pd.read_pickle(self.base_path + 'MM-SADA_Domain_Adaptation_Splits/'+source_dom1+"_"+split+".pkl")

            for _, line in train_file.iterrows():
                image = [source_dom1 + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                        line['stop_timestamp']]
                labels = line['verb_class']
                if int(labels) in source_1:
                    if int(labels) == 7:
                        labels = 3
                    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))

            source_dom2 = domain[1]
            train_file = pd.read_pickle(self.base_path + 'MM-SADA_Domain_Adaptation_Splits/'+source_dom2+"_"+split+".pkl")

            for _, line in train_file.iterrows():
                image = [source_dom2 + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                        line['stop_timestamp']]
                labels = line['verb_class']
                if int(labels) in source_2:
                    if int(labels) == 7:
                        labels = 3
                    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
        else:
            target_dom = domain[0]
            train_file = pd.read_pickle(self.base_path + 'MM-SADA_Domain_Adaptation_Splits/'+target_dom+"_"+split+".pkl")

            for _, line in train_file.iterrows():
                image = [target_dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                        line['stop_timestamp']]
                labels = line['verb_class']
                if int(labels) in target_all:
                    if int(labels) == 3:
                        labels = 7
                    elif int(labels) == 7:
                        labels = 3
                    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
                
        self.samples = data1
        self.cfg = cfg

    def __getitem__(self, index):
        video_path = self.base_path +'rgb/'+self.split + '/'+self.samples[index][0]

        filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))
        data = dict(
            frame_dir=video_path,
            total_frames=int(self.samples[index][2] - self.samples[index][1]),
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data = self.pipeline(data)

        label1 = self.samples[index][-1]

        audio_path = self.base_path + 'rgb/' + self.split + '/' + self.samples[index][0] + '.wav'
        samples, samplerate = sf.read(audio_path)

        duration = len(samples) / samplerate

        fr_sec = self.samples[index][3].split(':')
        hour1 = float(fr_sec[0])
        minu1 = float(fr_sec[1])
        sec1 = float(fr_sec[2])
        fr_sec = (hour1 * 60 + minu1) * 60 + sec1

        stop_sec = self.samples[index][4].split(':')
        hour1 = float(stop_sec[0])
        minu1 = float(stop_sec[1])
        sec1 = float(stop_sec[2])
        stop_sec = (hour1 * 60 + minu1) * 60 + sec1

        start1 = fr_sec / duration * len(samples)
        end1 = stop_sec / duration * len(samples)
        start1 = int(np.round(start1))
        end1 = int(np.round(end1))
        samples = samples[start1:end1]

        resamples = samples[:160000]
        while len(resamples) < 160000:
            resamples = np.tile(resamples, 10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        if self.split == 'train':
            noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
            spectrogram = spectrogram + noise
            start1 = np.random.choice(256 - self.interval, (1,))[0]
            spectrogram[start1:(start1 + self.interval), :] = 0

        return data, spectrogram.astype(np.float32), label1

    def __len__(self):
        return len(self.samples)

