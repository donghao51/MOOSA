<div align="center">

<h1>Towards Multimodal Open-Set Domain Generalization and Adaptation through Self-supervision</h1>

<div>
    <a href='https://sites.google.com/view/dong-hao/' target='_blank'>Hao Dong</a><sup>1</sup>&emsp;
    <a href='https://chatzi.ibk.ethz.ch/about-us/people/prof-dr-eleni-chatzi.html' target='_blank'>Eleni Chatzi</a><sup>1</sup>&emsp;
    <a href='https://people.epfl.ch/olga.fink?lang=en' target='_blank'>Olga Fink</a><sup>2</sup>
</div>
<div>
    <sup>1</sup>ETH Zurich, <sup>2</sup>EPFL
</div>


<div>
    <h4 align="center">
        • <a href="https://arxiv.org/abs/2407.01518" target='_blank'>ECCV 2024</a> •
    </h4>
</div>



<div style="text-align:center">
<img src="imgs/MOOSA.png"  width="80%" height="100%">
</div>

---

</div>


Our proposed MOOSA framework for Multimodal Open-Set Domain Generalization and Adaptation.

## Code
The code was tested using `Python 3.10.13`, `torch 2.3.1+cu121` and `NVIDIA GeForce RTX 3090`, more dependencies are in `requirement.txt`.

Environments:
```
mmcv-full 1.2.7
mmaction2 0.13.0
```
### EPIC-Kitchens Dataset
### Prepare

#### Download Pretrained Weights
1. Download Audio model [link](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), rename it as `vggsound_avgpool.pth.tar` and place under the `EPIC-rgb-flow-audio/pretrained_models` directory
   
2. Download SlowFast model for RGB modality [link](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) and place under the `EPIC-rgb-flow-audio/pretrained_models` directory
   
3. Download SlowOnly model for Flow modality [link](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) and place under the `EPIC-rgb-flow-audio/pretrained_models` directory

#### Download EPIC-Kitchens Dataset
```
bash download_script.sh 
```
Download Audio files [EPIC-KITCHENS-audio.zip](https://polybox.ethz.ch/index.php/s/PE2zIL99OWXQfMu).

Unzip all files and the directory structure should be modified to match:
<details>
<summary>Click for details...</summary>

```
├── MM-SADA_Domain_Adaptation_Splits
├── rgb
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01.wav
|   |   |   ├── P08_01
|   |   |   |     ├── frame_0000000000.jpg
|   |   |   |     ├── ...
|   |   |   ├── P08_02.wav
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3


├── flow
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01 
|   |   |   |   ├── u
|   |   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   |   ├── ...
|   |   |   |   ├── v
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3
```

</details>

### Video and Audio
<details>
<summary>Click for details...</summary>

```
cd EPIC-rgb-flow-audio
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 5 --mask_ratio 0.7 --entropy_min_weight 0.001 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.7 --entropy_min_weight 1.0 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.7 --entropy_min_weight 0.001 --datapath /path/to/EPIC-KITCHENS/
```

</details>

### Video and Flow
<details>
<summary>Click for details...</summary>

```
cd EPIC-rgb-flow-audio
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_flow -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --mask_ratio 0.7 --entropy_min_weight 0.001 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_flow -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.7 --entropy_min_weight 1.0 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_flow -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 15 --mask_ratio 0.7 --entropy_min_weight 0.001 --datapath /path/to/EPIC-KITCHENS/
```

</details>

### Flow and Audio
<details>
<summary>Click for details...</summary>


```
cd EPIC-rgb-flow-audio
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 25 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/EPIC-KITCHENS/
```

</details>

### Video and Flow and Audio
<details>
<summary>Click for details...</summary>


```
cd EPIC-rgb-flow-audio
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --mask_ratio 0.7 --entropy_min_weight 0.001 --jigsaw_num_splits 2 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.7 --entropy_min_weight 0.1 --jigsaw_num_splits 2 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.3 --entropy_min_weight 0.1 --jigsaw_num_splits 2  --datapath /path/to/EPIC-KITCHENS/
```

</details>



### HAC Dataset
This dataset can be downloaded at [link](https://polybox.ethz.ch/index.php/s/3F8ZWanMMVjKwJK).

Unzip all files and the directory structure should be modified to match:
<details>
<summary>Click for details...</summary>

```
HAC
├── human
|   ├── videos
|   |   ├── ...
|   ├── flow
|   |   ├── ...
|   ├── audio
|   |   ├── ...

├── animal
|   ├── videos
|   |   ├── ...
|   ├── flow
|   |   ├── ...
|   ├── audio
|   |   ├── ...

├── cartoon
|   ├── videos
|   |   ├── ...
|   ├── flow
|   |   ├── ...
|   ├── audio
|   |   ├── ...
```

</details>

Download the pretrained weights similar to EPIC-Kitchens Dataset and put under the `HAC-rgb-flow-audio/pretrained_models` directory.

### Video and Audio
<details>
<summary>Click for details...</summary>


```
cd HAC-rgb-flow-audio
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_audio -s 'animal' 'cartoon' -t 'human' --lr 1e-4 --bsz 16 --nepochs 5 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_audio -s 'human' 'cartoon' -t 'animal' --lr 1e-4 --bsz 16  --nepochs 10 --mask_ratio 0.7 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_audio -s 'human' 'animal' -t 'cartoon' --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```

</details>

### Video and Flow
<details>
<summary>Click for details...</summary>


```
cd HAC-rgb-flow-audio
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_flow -s 'animal' 'cartoon' -t 'human' --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_flow -s 'human' 'cartoon' -t 'animal' --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_flow -s 'human' 'animal' -t 'cartoon' --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```

</details>

### Flow and Audio
<details>
<summary>Click for details...</summary>


```
cd HAC-rgb-flow-audio
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_flow --use_audio -s 'animal' 'cartoon' -t 'human' --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_flow --use_audio -s 'human' 'cartoon' -t 'animal' --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.3 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_flow --use_audio -s 'human' 'animal' -t 'cartoon' --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.7 --entropy_min_weight 0.001 --datapath /path/to/HAC/
```

</details>

### Video and Flow and Audio
<details>
<summary>Click for details...</summary>


```
cd HAC-rgb-flow-audio
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_flow --use_audio -s 'animal' 'cartoon' -t 'human' --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.7 --entropy_min_weight 0.001 --jigsaw_num_splits 2 --datapath /path/to/HAC/
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_flow --use_audio -s 'human' 'cartoon' -t 'animal' --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.7 --entropy_min_weight 0.001 --jigsaw_num_splits 2 --jigsaw_samples 64 --datapath /path/to/HAC/
```
```
python train_video_flow_audio_HAC_MOOSA.py --use_video --use_flow --use_audio -s 'human' 'animal' -t 'cartoon' --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.7 --alpha_trans 0.5 --entropy_min_weight 0.001 --jigsaw_num_splits 2 --datapath /path/to/HAC/
```

</details>

## Multimodal Open-Set Domain Adaptation
### Video and Audio
<details>
<summary>Click for details...</summary>

```
cd EPIC-rgb-flow-audio
```
```
python train_video_audio_EPIC_MOOSA_OSDA.py -s D1 D2 -t D2 --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.3 --target_filter_thr 0.5 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_audio_EPIC_MOOSA_OSDA.py -s D1 D3 -t D3 --lr 1e-4 --bsz 16 --nepochs 10 --mask_ratio 0.7 --target_filter_thr 0.5 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_audio_EPIC_MOOSA_OSDA.py -s D2 D1 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --mask_ratio 0.7 --target_filter_thr 0.5 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_audio_EPIC_MOOSA_OSDA.py -s D2 D3 -t D3 --lr 1e-4 --bsz 16 --nepochs 15 --mask_ratio 0.7 --target_filter_thr 0.5 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_audio_EPIC_MOOSA_OSDA.py -s D3 D1 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --mask_ratio 0.7 --target_filter_thr 0.3 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_audio_EPIC_MOOSA_OSDA.py -s D3 D2 -t D2 --lr 1e-4 --bsz 16 --nepochs 15 --mask_ratio 0.7 --target_filter_thr 0.5 --datapath /path/to/EPIC-KITCHENS/
```


</details>

## Contact
If you have any questions, please send an email to donghaospurs@gmail.com

## Citation

If you find our work useful in your research please consider citing our [paper](https://arxiv.org/abs/2407.01518):

```
@inproceedings{dong2024moosa,
    title={Towards Multimodal Open-Set Domain Generalization and Adaptation through Self-supervision},
    author={Dong, Hao and Chatzi, Eleni and Fink, Olga},
    booktitle={European Conference on Computer Vision},
    year={2024}
}
```

## Related Projects

[SimMMDG](https://github.com/donghao51/SimMMDG): A Simple and Effective Framework for Multi-modal Domain Generalization

[MultiOOD](https://github.com/donghao51/MultiOOD): Scaling Out-of-Distribution Detection for Multiple Modalities



## Acknowledgement

Many thanks to the excellent open-source projects [SimMMDG](https://github.com/donghao51/SimMMDG) and [DomainAdaptation](https://github.com/xiaobai1217/DomainAdaptation).
