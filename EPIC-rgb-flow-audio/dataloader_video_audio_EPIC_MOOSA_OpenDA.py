from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

source_1 = [0, 1, 2, 4, 5, 6, 7]
source_2 = [0, 1, 2, 4, 5, 6, 7]
source_all = [0, 1, 2, 4, 5, 6, 7]
target_all = [0, 1, 2, 3, 4, 5, 6, 7]

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', domain_name='source', domain=['D1'], modality='rgb', cfg=None, use_audio=True, sample_dur=10, datapath='/scratch/project_2000948/data/haod/EPIC-KITCHENS/'):
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.use_audio = use_audio
        self.interval = 9
        self.sample_dur = sample_dur

        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            target_train_pipeline = cfg.data.train.pipeline
            self.train_pipeline = Compose(train_pipeline)
            self.target_train_pipeline = Compose(target_train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.val_pipeline = Compose(val_pipeline)

        data1 = []
        target_data1 = []
        class_dict = {}
        if domain_name == 'source' and split == 'train':
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
                    if line['verb'] not in list(class_dict.keys()):
                        class_dict[line['verb']] = line['verb_class']

            target_dom = domain[1]
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
                    target_data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
                    if line['verb'] not in list(class_dict.keys()):
                        class_dict[line['verb']] = line['verb_class']
        elif domain_name == 'source' and split == 'test':
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
                    if line['verb'] not in list(class_dict.keys()):
                        class_dict[line['verb']] = line['verb_class']
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
                    if line['verb'] not in list(class_dict.keys()):
                        class_dict[line['verb']] = line['verb_class']


        self.class_dict = class_dict
        self.samples = data1
        self.target_samples = target_data1
        self.cfg = cfg
        self.target_cfg = cfg

    def __getitem__(self, index):
        target_index = index 
        if len(self.target_samples) > 0:
            if len(self.samples) > len(self.target_samples):
                if target_index >= len(self.target_samples):
                    target_index = target_index % len(self.target_samples)
            elif len(self.samples) < len(self.target_samples):
                if index >= len(self.samples):
                    index = index % len(self.samples)

        video_path = self.base_path +'rgb/'+self.split + '/'+self.samples[index][0]
        if self.split == 'train':
            target_video_path = self.base_path +'rgb/'+self.split + '/'+self.target_samples[target_index][0]

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
            data = self.train_pipeline(data)

            target_filename_tmpl = self.target_cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
            target_modality = self.target_cfg.data.train.get('modality', 'RGB')

            target_start_index = self.target_cfg.data.train.get('start_index', int(self.target_samples[target_index][1]))
            target_data = dict(
                frame_dir=target_video_path,
                total_frames=int(self.target_samples[target_index][2] - self.target_samples[target_index][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=target_start_index,
                filename_tmpl=target_filename_tmpl,
                modality=target_modality)
            target_data = self.target_train_pipeline(target_data)
        else:
            filename_tmpl = self.cfg.data.val.get('filename_tmpl', 'frame_{:010}.jpg')
            modality = self.cfg.data.val.get('modality', 'RGB')
            start_index = self.cfg.data.val.get('start_index', int(self.samples[index][1]))
            data = dict(
                frame_dir=video_path,
                total_frames=int(self.samples[index][2] - self.samples[index][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data = self.val_pipeline(data)
        label1 = self.samples[index][-1]

        if self.use_audio is True:
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

            dur = int(self.sample_dur * 16000)
            resamples = samples[:dur]
            while len(resamples) < dur:
                resamples = np.tile(resamples, 10)[:dur]

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

                # target domain
                audio_path = self.base_path + 'rgb/' + self.split + '/' + self.target_samples[target_index][0] + '.wav'
                samples, samplerate = sf.read(audio_path)

                duration = len(samples) / samplerate

                fr_sec = self.target_samples[target_index][3].split(':')
                hour1 = float(fr_sec[0])
                minu1 = float(fr_sec[1])
                sec1 = float(fr_sec[2])
                fr_sec = (hour1 * 60 + minu1) * 60 + sec1

                stop_sec = self.target_samples[target_index][4].split(':')
                hour1 = float(stop_sec[0])
                minu1 = float(stop_sec[1])
                sec1 = float(stop_sec[2])
                stop_sec = (hour1 * 60 + minu1) * 60 + sec1

                start1 = fr_sec / duration * len(samples)
                end1 = stop_sec / duration * len(samples)
                start1 = int(np.round(start1))
                end1 = int(np.round(end1))
                samples = samples[start1:end1]

                dur = int(self.sample_dur * 16000)
                resamples = samples[:dur]
                while len(resamples) < dur:
                    resamples = np.tile(resamples, 10)[:dur]

                resamples[resamples > 1.] = 1.
                resamples[resamples < -1.] = -1.
                frequencies, times, target_spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
                target_spectrogram = np.log(target_spectrogram + 1e-7)

                mean = np.mean(target_spectrogram)
                std = np.std(target_spectrogram)
                target_spectrogram = np.divide(target_spectrogram - mean, std + 1e-9)

                noise = np.random.uniform(-0.05, 0.05, target_spectrogram.shape)
                target_spectrogram = target_spectrogram + noise
                start1 = np.random.choice(256 - self.interval, (1,))[0]
                target_spectrogram[start1:(start1 + self.interval), :] = 0

        if self.split == 'train':
            return data, spectrogram.astype(np.float32), label1, target_data, target_spectrogram.astype(np.float32)
        else:
            return data, spectrogram.astype(np.float32), label1, 0, 0

    def __len__(self):
        return max(len(self.samples), len(self.target_samples))

