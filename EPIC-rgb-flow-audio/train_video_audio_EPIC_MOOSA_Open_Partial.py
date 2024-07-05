from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_audio_EPIC_MOOSA_Open_Partial import EPICDOMAIN
import torch.nn.functional as F
from losses import SupConLoss
from torch.distributions import Categorical
import itertools

source_1 = [0, 1, 2, 5, 7]
source_2 = [0, 1, 4, 6, 7]
source_all = [0, 1, 2, 4, 5, 6, 7]
target_all = [0, 1, 3, 5, 6]

def train_one_step(clip, labels, spectrogram):
    labels = labels.cuda()
    clip = clip['imgs'].cuda().squeeze(1)
    spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)  
        v_feat = (x_slow.detach(), x_fast.detach())  
        _, audio_feat, _ = audio_model(spectrogram)

    v_feat = model.module.backbone.get_predict(v_feat)
    predict1, v_emd = model.module.cls_head(v_feat)
    v_dim = int(v_emd.shape[1] / 2)
    entropyp = Categorical(probs = nn.Softmax(dim=1)(predict1)).entropy().reshape(-1,1)
    output_loss1 = criterion(predict1, labels)
    video_parts = torch.split(v_emd, v_emd.shape[1] // args.jigsaw_num_splits, dim=1)
    output_u_v = F.softmax(predict1, 1)
    loss_u_v = (-output_u_v * torch.log(output_u_v + 1e-5)).sum(1).mean()

    audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
    a_dim = int(audio_emd.shape[1] / 2)
    entropya = Categorical(probs = nn.Softmax(dim=1)(audio_predict)).entropy().reshape(-1,1)
    output_loss2 = criterion(audio_predict, labels)
    audio_parts = torch.split(audio_emd, audio_emd.shape[1] // args.jigsaw_num_splits, dim=1)
    output_u_a = F.softmax(audio_predict, 1)
    loss_u_a = (-output_u_a * torch.log(output_u_a + 1e-5)).sum(1).mean()

    feat = torch.cat((v_emd, audio_emd), dim=1)
    parts = video_parts + audio_parts 

    predict = mlp_cls(feat)
    output_loss4 = criterion(predict, labels)
    entropypa = Categorical(probs = nn.Softmax(dim=1)(predict)).entropy().reshape(-1,1)
    output_u = F.softmax(predict, 1)
    loss_u = (-output_u * torch.log(output_u + 1e-5)).sum(1).mean()

    # Entropy Weighting
    entropy = -torch.cat((entropyp, entropya, entropypa), 1)
    output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)
    loss = torch.mean(output[:,0]*output_loss1+output[:,1]*output_loss2+output[:,2]*output_loss4)
    loss_ent_min = (loss_u + loss_u_v + loss_u_a) * args.entropy_min_weight / 3

    # Multimodal Jigsaw Puzzles
    all_combinations = list(itertools.permutations(parts, len(parts)))
    all_combinations = [all_combinations[ji] for ji in jigsaw_indices]
    jigsaw_labels = []
    combinations = []
    for label, all_parts in enumerate(all_combinations):
        concatenated = torch.cat(all_parts, dim=1)
        jigsaw_labels.append(torch.tensor([label]).repeat(concatenated.shape[0], 1))
        combinations.append(concatenated)
    combinations = torch.cat(combinations, dim=0)
    jigsaw_labels = torch.cat(jigsaw_labels, dim=0).squeeze(1).type(torch.LongTensor).cuda()
    predict_jigsaw = jigsaw_cls(combinations)
    loss_jigsaw = nn.CrossEntropyLoss()(predict_jigsaw, jigsaw_labels)
    loss = loss + loss_jigsaw*args.jigsaw_ratio

    # Entropy Minimization
    loss = loss + loss_ent_min

    # Masked Cross-modal Translation 
    mask_v = torch.rand_like(v_emd) < args.mask_ratio
    v_emd_masked = v_emd.clone()  
    v_emd_masked[mask_v] = 0 

    mask_a = torch.rand_like(audio_emd) < args.mask_ratio
    audio_emd_masked = audio_emd.clone()  
    audio_emd_masked[mask_a] = 0 

    a_emd_t = mlp_v2a(v_emd_masked)
    v_emd_t = mlp_a2v(audio_emd_masked)
    a_emd_t = a_emd_t/torch.norm(a_emd_t, dim=1, keepdim=True)
    v_emd_t = v_emd_t/torch.norm(v_emd_t, dim=1, keepdim=True)
    v2a_loss = torch.mean(torch.norm(a_emd_t-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
    a2v_loss = torch.mean(torch.norm(v_emd_t-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
    loss = loss + args.alpha_trans*(v2a_loss + a2v_loss)/2

    # Supervised Contrastive Learning
    v_emd_proj = v_proj(v_emd[:, :v_dim])
    a_emd_proj = a_proj(audio_emd[:, :a_dim])
    emd_proj = torch.stack([v_emd_proj, a_emd_proj], dim=1)

    loss_contrast = criterion_contrast(emd_proj, labels)
    loss = loss + args.alpha_contrast*loss_contrast
  
    # Feature Splitting with Distance
    loss_e = - F.mse_loss(v_emd[:, :v_dim], v_emd[:, v_dim:]) - F.mse_loss(audio_emd[:, :a_dim], audio_emd[:, a_dim:])
    
    loss = loss + args.explore_loss_coeff * loss_e/2

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predict, loss

def validate_one_step(clip, labels, spectrogram):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
    
    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip) 
        v_feat = (x_slow.detach(), x_fast.detach())  

        v_feat = model.module.backbone.get_predict(v_feat)
        predict1, v_emd = model.module.cls_head(v_feat)
        _, audio_feat, _ = audio_model(spectrogram)
        audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

        feat = torch.cat((v_emd, audio_emd), dim=1)

        predict = mlp_cls(feat)

    loss = torch.mean(criterion(predict, labels))

    return predict, loss

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)

class EncoderTrans(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat

class EncoderJigsaw(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderJigsaw, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)

class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
    parser.add_argument('--datapath', type=str, default='/path/to/EPIC-KITCHENS/',
                        help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=15)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--alpha_trans', type=float, default=0.1,
                        help='alpha_trans')
    parser.add_argument("--trans_hidden_num", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temp')
    parser.add_argument('--alpha_contrast', type=float, default=3.0,
                        help='alpha_contrast')
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument('--explore_loss_coeff', type=float, default=0.7,
                        help='explore_loss_coeff')
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestAcc', type=float, default=0,
                        help='BestAcc')
    parser.add_argument('--BestTestAcc', type=float, default=0,
                        help='BestTestAcc')
    parser.add_argument('--BestTestHscore', type=float, default=0,
                        help='BestTestHscore')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--entropy_weight_temp', type=float, default=1.0,
                        help='entropy_weight_temp')
    parser.add_argument('--entropy_min_weight', type=float, default=0.001,
                        help='entropy_min_weight')
    parser.add_argument('--jigsaw_ratio', type=float, default=1.0,
                        help='jigsaw_ratio')
    parser.add_argument("--jigsaw_num_splits", type=int, default=4)
    parser.add_argument("--jigsaw_samples", type=int, default=128)
    parser.add_argument("--jigsaw_hidden", type=int, default=512)
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                        help='mask_ratio')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    jigsaw_indices = random.sample(range(np.math.factorial(2*args.jigsaw_num_splits)), args.jigsaw_samples)
    print('jigsaw_indices: ', jigsaw_indices)

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    num_class = len(source_all)
    input_dim = 0

    cfg = None
    
    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(2304, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    v_proj = ProjectHead(input_dim=1152, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
    input_dim = input_dim + 2304

    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar")
    audio_model.load_state_dict(checkpoint['model_state_dict'])
    audio_model = audio_model.cuda()
    audio_model.eval()

    audio_cls_model = AudioAttGenModule()
    audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    audio_cls_model.fc = nn.Linear(512, num_class)
    audio_cls_model = audio_cls_model.cuda()

    a_proj = ProjectHead(input_dim=256, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
    input_dim = input_dim + 512

    mlp_cls = Encoder(input_dim=input_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()

    jigsaw_cls = EncoderJigsaw(input_dim=input_dim, out_dim=args.jigsaw_samples, hidden=args.jigsaw_hidden)
    jigsaw_cls = jigsaw_cls.cuda()

    mlp_v2a = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=512).cuda()
    mlp_a2v = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2304).cuda()

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path_model = "models/"
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)

    log_name = "log%s2%s_MOOSA_open_partial_video_audio"%(args.source_domain, args.target_domain)

    log_name = log_name + '_entropy_min' + '_' + str(args.entropy_min_weight)
    log_name = log_name + '_entropy_weight' + '_' + str(args.entropy_weight_temp)
    log_name = log_name + '_trans_mask_%s_'%(str(args.trans_hidden_num)) + str(args.alpha_trans)+ '_' + str(args.mask_ratio)
    log_name = log_name + '_jigsaw_' + str(args.jigsaw_num_splits) + '_' + str(args.jigsaw_samples) + '_' + str(args.jigsaw_ratio)+ '_' + str(args.jigsaw_hidden)

    log_name = log_name + args.appen
    log_path = base_path + log_name + '.csv'
    print(log_path)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.cuda()
    batch_size = args.bsz

    criterion_contrast = SupConLoss(temperature=args.temp)
    criterion_contrast = criterion_contrast.cuda()

    params = list(mlp_cls.parameters())
    params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(v_proj.parameters())

    params = params + list(audio_cls_model.parameters()) + list(a_proj.parameters())

    params = params + list(mlp_v2a.parameters())+list(mlp_a2v.parameters())

    params = params + list(jigsaw_cls.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    
    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc
    BestTestHscore = args.BestTestHscore
    BestTestOS = 0
    BestTestUNK = 0

    if args.resumef:
        resume_file = base_path_model + log_name + '.pt'
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch']+1
    
        BestLoss = checkpoint['BestLoss']
        BestEpoch = checkpoint['BestEpoch']
        BestAcc = checkpoint['BestAcc']
        BestTestAcc = checkpoint['BestTestAcc']
        BestTestHscore = checkpoint['BestTestHscore']

        model.load_state_dict(checkpoint['model_state_dict'])
        v_proj.load_state_dict(checkpoint['v_proj_state_dict'])
        audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
        audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
        a_proj.load_state_dict(checkpoint['a_proj_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        mlp_v2a.load_state_dict(checkpoint['mlp_v2a_state_dict'])
        mlp_a2v.load_state_dict(checkpoint['mlp_a2v_state_dict'])
        mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
        jigsaw_cls.load_state_dict(checkpoint['jigsaw_cls_state_dict'])
    else:
        print("Training From Scratch ..." )
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)

    train_dataset = EPICDOMAIN(split='train', domain_name='source', domain=args.source_domain, cfg=cfg, datapath=args.datapath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)

    val_dataset = EPICDOMAIN(split='test', domain_name='source', domain=args.source_domain, cfg=cfg, datapath=args.datapath)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)

    test_dataset = EPICDOMAIN(split='test', domain_name='target', domain=args.target_domain, cfg=cfg, datapath=args.datapath)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    with open(log_path, "a") as f:
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                mlp_cls.train(split == 'train')
                jigsaw_cls.train(split == 'train')
                model.train(split == 'train')
                v_proj.train(split == 'train')
                audio_cls_model.train(split == 'train')
                a_proj.train(split == 'train')
                mlp_v2a.train(split == 'train')
                mlp_a2v.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    if split=='test':
                        output_sum = []
                        target_sum = []

                        with torch.no_grad():
                            for (i, (clip, spectrogram, labels)) in enumerate(dataloaders[split]):
                                target = labels
                                clip = clip['imgs'].cuda().squeeze(1)
                                x_slow, x_fast = model.module.backbone.get_feature(clip)  
                                v_feat = (x_slow.detach(), x_fast.detach()) 
                                v_feat = model.module.backbone.get_predict(v_feat)
                                predict1, v_emd = model.module.cls_head(v_feat)
                                spectrogram = spectrogram.unsqueeze(1).cuda()
                                _, audio_feat, _ = audio_model(spectrogram)
                                audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

                                feat = torch.cat((v_emd, audio_emd), dim=1)
                                predict = mlp_cls(feat)

                                outlier_flag = (target > (num_class - 1)).float()
                                target = target * (1 - outlier_flag) + num_class * outlier_flag
                                target = target.long()
                                output_sum.append(predict)
                                target_sum.append(target)
                        output_sum = torch.cat(output_sum)
                        target_sum = torch.cat(target_sum)

                        tsm_output = F.softmax(output_sum, dim=1)
                        outlier_indis, max_index = torch.max(tsm_output, 1)
                        thd_min = torch.min(outlier_indis)
                        thd_max = torch.max(outlier_indis)
                        outlier_range = [thd_min + (thd_max - thd_min) * k / 9 for k in range(10)]

                        best_overall_acc = 0.0
                        best_thred_acc = 0.0
                        best_overall_Hscore = 0.0
                        best_thred_Hscore = 0.0
                        best_acc_insider = 0.0
                        best_acc_outsider = 0.0

                        for outlier_thred in outlier_range:
                            outlier_pred = (outlier_indis < outlier_thred).float()
                            outlier_pred = outlier_pred.view(-1, 1)
                            output = torch.cat((tsm_output, outlier_pred.cuda()), dim=1)

                            _, predict = torch.max(output.detach().cpu(), dim=1)
                            overall_acc = (predict == target_sum).sum().item() / target_sum.shape[0]

                            indices_outsider = torch.where(target_sum == num_class)[0]
                            indices_insider = torch.where(target_sum != num_class)[0]
                            acc_insider = (predict[indices_insider] == target_sum[indices_insider]).sum().item() / target_sum[indices_insider].shape[0]
                            acc_outsider = (predict[indices_outsider] == target_sum[indices_outsider]).sum().item() / target_sum[indices_outsider].shape[0]
                            overall_Hscore = 2.0 * acc_insider * acc_outsider / (acc_insider + acc_outsider)

                            if overall_acc > best_overall_acc:
                                best_overall_acc = overall_acc
                                best_thred_acc = outlier_thred
                            if overall_Hscore > best_overall_Hscore:
                                best_overall_Hscore = overall_Hscore
                                best_thred_Hscore = outlier_thred
                                best_acc_insider = acc_insider
                                best_acc_outsider = acc_outsider

                    else:
                        for (i, (clip, spectrogram, labels)) in enumerate(dataloaders[split]):
                            if split=='train':
                                predict1, loss = train_one_step(clip, labels, spectrogram)
                            else:
                                predict1, loss = validate_one_step(clip, labels, spectrogram)

                            total_loss += loss.item() * batch_size
                            _, predict = torch.max(predict1.detach().cpu(), dim=1)

                            acc1 = (predict == labels).sum().item()
                            acc += int(acc1)
                            count += predict1.size()[0]
                            pbar.set_postfix_str(
                                "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                    loss.item(),
                                                                                                    acc / float(count)))
                            pbar.update()

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)
                            
                    if split == 'test':
                        if currentvalAcc >= BestAcc:
                            BestTestAcc = best_overall_acc
                            BestTestHscore = best_overall_Hscore
                            BestTestOS = best_acc_insider
                            BestTestUNK = best_acc_outsider
                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'BestTestHscore': BestTestHscore,
                                    'optimizer': optim.state_dict(),
                                }
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                                save['jigsaw_cls_state_dict'] = jigsaw_cls.state_dict()
                                
                                save['v_proj_state_dict'] = v_proj.state_dict()
                                save['model_state_dict'] = model.state_dict()
                            
                                save['a_proj_state_dict'] = a_proj.state_dict()
                                save['audio_model_state_dict'] = audio_model.state_dict()
                                save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                            
                                save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()

                                torch.save(save, base_path_model + log_name + '_best_%s.pt'%(str(epoch_i)))

                        if args.save_checkpoint:
                            save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'BestTestHscore': BestTestHscore,
                                    'optimizer': optim.state_dict(),
                                }
                            save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                            save['jigsaw_cls_state_dict'] = jigsaw_cls.state_dict()
                            
                            save['v_proj_state_dict'] = v_proj.state_dict()
                            save['model_state_dict'] = model.state_dict()
                            save['a_proj_state_dict'] = a_proj.state_dict()
                            save['audio_model_state_dict'] = audio_model.state_dict()
                            save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                        
                            save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                            save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()

                            torch.save(save, base_path_model + log_name + '.pt')
                        
                    if split == 'test':
                        f.write("{},{},{},{},{},{},{},{}\n".format(epoch_i, split, best_thred_acc, best_thred_Hscore, best_overall_acc, best_acc_insider, best_acc_outsider, best_overall_Hscore))
                    else:
                        f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)
                    print('BestTestHscore ', BestTestHscore)
                    print('BestTestOS ', BestTestOS)
                    print('BestTestUNK ', BestTestUNK)
                    
                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{},OS,{},UNK,{},BestTestHscore,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc, BestTestOS, BestTestUNK, BestTestHscore))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{},OS,{},UNK,{},BestTestHscore,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc, BestTestOS, BestTestUNK, BestTestHscore))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)
        print('BestTestHscore ', BestTestHscore)

    f.close()
