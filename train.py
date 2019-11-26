from src.generator import Generator
from src.discriminator import Discriminator
from honk.utils.model import *
from torchsummary import summary
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import argparse

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)


def train(args):
    config = SpeechDataset.default_config()
    config["wanted_words"] = "yes no marvin left right".split()
    config["data_folder"] = "/content/data"
    config["cache_size"] = 32768
    config["batch_size"] = 64
    train_set, dev_set, test_set = SpeechDataset.splits(config)

    train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
        collate_fn=train_set.collate_fn)
    dev_loader = data.DataLoader(
        dev_set,
        batch_size=min(len(dev_set), 16),
        shuffle=True,
        collate_fn=dev_set.collate_fn)
    test_loader = data.DataLoader(
        test_set,
        batch_size=min(len(test_set), 16),
        shuffle=True,
        collate_fn=test_set.collate_fn)

    gen = Generator()
    disc = Discriminator()
    optim_gen = torch.optim.Adam(lr=1e-3,params=gen.parameters(),weight_decay=1e-3)
    optim_disc = torch.optim.Adam(lr=1e-3,params=disc.parameters(),weight_decay=1e-3)

    start_epoch = 0

    if args.weights_path is not None:
        weights_dict = torch.load(args.weights_path)
        start_epoch = weights_dict['epoch']+1
        gen.load_state_dict(weights_dict['gen_state_dict'])
        disc.load_state_dict(weights_dict['disc_state_dict'])
        optim_gen.load_state_dict(weights_dict['optim_gen_state_dict'])
        optim_disc.load_state_dict(weights_dict['optim_disc_state_dict'])
    else:
         gen_state_dict = gen.state_dict()
         for key in gen_state_dict.keys():
             if gen_state_dict[key].dim() >= 2:
                 torch.nn.init.xavier_normal_(gen_state_dict[key],1e-2)
             else:
                 if key[-4:] == 'bias':
                     torch.nn.init.zeros_(gen_state_dict[key])
                 else:
                     torch.nn.init.ones_(gen_state_dict[key])
        gen.load_state_dict(gen_state_dict)

    model_config = dict(dropout_prob=0.5, height=128, width=40, n_labels=7, n_feature_maps1=64,
            n_feature_maps2=64, conv1_size=(20, 8), conv2_size=(10, 4), conv1_pool=(2, 2), conv1_stride=(1, 1),
            conv2_stride=(1, 1), conv2_pool=(1, 1), tf_variant=True)
    kws_model = SpeechModel(model_config)
    kws_model.load(args.kws_model_path)

    dct_filters = torch.from_numpy(np.load('dct_filter.npy').astype(np.float32))
    if torch.cuda.is_available():
        dct_filters = dct_filters.cuda()

    num_epochs = args.epochs
    c = args.c
    alpha = args.alpha
    beta = args.beta
    mean = torch.load('spectrogram_mean.pkl')
    std = torch.load('spectrogram_std.pkl')

    for epoch in range(start_epoch,num_epochs):
        gen.train()
        disc.train()

        for step, sample in enumerate(train_loader):
            inp, labels = sample
            if torch.cuda.is_available():
                inp = inp.cuda()
                labels = labels.cuda()

            gen_noise = gen(((inp-mean)/std).permute(0,2,1))
            gen_noise[:,:,101:] = torch.zeros(gen_noise.shape[0],128,27)
            noise_score = disc((((inp-mean)/std).permute(0,2,1) + gen_noise).unsqueeze(1))
            inp_score = disc(((inp-mean)/std).permute(0,2,1).unsqueeze(1))

            kws_inp = inp + gen_noise.permute(0,2,1)*std
            kws_inp = kws_inp[:,:101,:].reshape(-1,128,101)
            kws_inp_clone = kws_inp.clone()
            kws_inp_clone[kws_inp_clone>0] = torch.log(kws_inp[kws_inp>0])
            mfcc_feat = torch.matmul(dct_filters, kws_inp_clone).permute(0,2,1)
            mfcc_feat = F.pad(mfcc_feat,(0,0,0,128-mfcc_feat.shape[1]))

            kws_out = nn.Softmax(dim=1)(kws_model(mfcc_feat))

            # Optimise Generator
            optim_gen.zero_grad()
            loss_gen = - noise_score.log().mean()
            loss_adv = kws_out.gather(1,labels.view(-1,1)).mean()
            loss_hinge = nn.ReLU()((gen_noise*std).norm(p=2, dim=(1,2)) - c).mean()
            loss_gen_total = loss_gen + alpha*loss_hinge + beta*loss_adv
            loss_gen_total.backward(retain_graph=True)
            optim_gen.step()
            print("Epoch : ", epoch, " , Step : ", step)
            print("Generator Loss", loss_gen)
            print("Loss Adv", loss_adv)
            print("Loss Hinge", loss_hinge)

            # Optimise Discriminator
            optim_disc.zero_grad()
            loss_disc = -(inp_score.log().mean() + (1-noise_score).log().mean())
            loss_disc.backward()
            optim_disc.step()
            print("Discriminator Loss", loss_disc)

            print("======================================")

        weights_dict = {}
        weights_dict['epoch'] = epoch
        weights_dict['gen_state_dict'] = gen.state_dict()
        weights_dict['disc_state_dict'] = disc.state_dict()
        weights_dict['optim_gen_state_dict'] = optim_gen.state_dict()
        weights_dict['optim_disc_state_dict'] = optim_disc.state_dict()
        weights_dict_path = args.save_folder_path+'/epoch{}.weights'.format(epoch)
        torch.save(weights_dict,weights_dict_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path",type=str,default=None,help="Path to the saved weights")
    parser.add_argument("--num_epochs",type=int,help="Number of epochs to train on")
    parser.add_argument("--c",type=float,help="Value of hyperparameter c to be used for training")
    parser.add_argument("--alpha",type=float,help="Value of hyperparameter alpha to be used for training")
    parser.add_argument("--beta",type=float,help="Value of hyperparameter beta to be used for training")
    parser.add_argument("--kws_model_path",type=str,help="Path to the KWS model for which adversarial examples are to be generated")
    parser.add_argument("--save_folder_path",type=str,help="Path to the folder in which weights are to be saved")
    args = parser.parse_args()
    train(args)
