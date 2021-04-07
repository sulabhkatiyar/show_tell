import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention_choice
from datasets import *
from utils import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
import json
import argparse

# Data parameters
data_folder = 'path_to_dataset_folder'  # folder with data files saved by create_input_files.py
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
dataset='flickr30k'

# Model parameters

emb_dim = 512  # dimension of word embeddings, have to change it to final embeddings size if using ensembles
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 30  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu1, best_bleu2, best_bleu3, best_bleu4 = 0.,0.,0.,0.  # BLEU scores right now
guiding_bleu= 1 # 1: BLEU 1, 2: BLEU-2, 3: BLEU-3, 4: BLEU4 #THE BLEU METRIC USED TO GUIDE THE PROCESS
print_freq = 500  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None # path to checkpoint 'BEST_checkpoint_flickr8k_5_cap_per_img_1_min_word_freq.pth.tar', None if none

# The following is number of feature maps generated by encoder. Change the 'encoder_choice' variable in the models.py file too
encoder_dim=2048  #number of feature maps generated by encoder; 4032 for NasNetlarge, 1536 for InceptionResNetv2/Inception, 2048 for Resnet152/resnet50/resnet101/resnext/xception/senet154/polynet; 
                  #256 for Alexnet; 512 for SqueezeNet/resnet18/resnet34/vgg16/vgg13/vgg19/vgg11; 1920 for Densenet; 1024 for googlenet/shufflenet; 1280 for mobilenet/mnasnet; 1056 for nasnetmobile
                  #2688 for dpn131; 4320 for pnasnet5large; 1024 for densenet121
    
use_image_transform=False
choice = 0


def main():
    """
    Training and validation.
    """

    global best_bleu1, best_bleu2, best_bleu3, best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder
    global emb_dim, encoder_dim, data_name, word_map, guiding_bleu, embeddings_file, emb_folder, use_embeddings_ensemble, choice

    # Remove previous checkpoints if they exist in same directory 
    if checkpoint == None:
        if 'BEST_checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar' in os.listdir(os.getcwd()):
            os.remove('BEST_checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar')
        if 'checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar' in os.listdir(os.getcwd()):
            os.remove('checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar')


    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention_choice(embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       encoder_dim=encoder_dim,
                                       dropout=dropout, choice=choice)
            
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)            
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu_scores'][3]
        best_bleu3 = checkpoint['bleu_scores'][2]
        best_bleu2 = checkpoint['bleu_scores'][1]
        best_bleu1 = checkpoint['bleu_scores'][0]
        decoder = checkpoint['decoder']
        
        decoder_optimizer = checkpoint['decoder_optimizer']

        encoder = checkpoint['encoder']
        # encoder_optimizer = checkpoint['encoder_optimizer']
        # if fine_tune_encoder is True and encoder_optimizer is None:
        if fine_tune_encoder is True:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
        else:
            encoder_optimizer = checkpoint['encoder_optimizer']


    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    horizontal=transforms.RandomHorizontalFlip(p=0.5)
    vertical= transforms.RandomVerticalFlip(p=0.5)
    totensor=transforms.ToTensor()
    topil=transforms.ToPILImage()

    transforms_to_apply=[horizontal]
    transforms_list=[topil] + transforms_to_apply + [totensor] if use_image_transform else []

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose(transforms_list + [normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)    #other transforms are topil,horizontal,vertical,totensor,

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_learning_rate(decoder_optimizer, 0.95)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.9)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu1,recent_bleu2,recent_bleu3,recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        #CHANGE IF OTHER BLEU SCORES ARE GUIDING THE TRAINING 
        # Check if there was an improvement
        if guiding_bleu==4:
            is_best = recent_bleu4 > best_bleu4 
        elif guiding_bleu==3:
            is_best = recent_bleu3 > best_bleu3
        elif guiding_bleu==2:
            is_best = recent_bleu2 > best_bleu2
        elif guiding_bleu==1:
            is_best = recent_bleu1 > best_bleu1
        
        best_bleu1 = max(recent_bleu1, best_bleu1)
        best_bleu2 = max(recent_bleu2, best_bleu2)
        best_bleu3 = max(recent_bleu3, best_bleu3)
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        
        bleu_list=[recent_bleu1,recent_bleu2,recent_bleu3,recent_bleu4]

        # Save validation set loss for each epoch in the file
        with open('validation_logs.txt', 'a') as vl:
            if epoch == 0:
                vl.write('\n\nThe dataset is {}. \nThe BLEU scores for epoch {} are {}.\n'.format(data_name,epoch,bleu_list))
            else:
                vl.write('The BLEU scores for epoch {} are {}.\n'.format(epoch, bleu_list))

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, bleu_list, is_best)

    #New additions from new_utils.py
    run_number=get_run_num() if fine_tune_encoder else get_run_num()+1  
    fine_tune='Yes' if fine_tune_encoder else 'No'
    encoder_lrate= encoder_lr if fine_tune_encoder else 0
    # The following line of code writes a string to file 'run_logs.txt' which contains information about hyperparameter settings, transforms
    # and datasets.
    save_string_to_run_logs('For the run {}, with dataset {}, encoder_fine_tune={}, the maximum BLEU scores are {}, {}, {}, {}.\n'
    'The model was trained for {} epochs and trained till epoch number {}.\n'
    'The number of units in embedding layer,attention layer and decoder are {},{} and {} respectively and dropout was {}.\n'
    'Encoder and decoder learning rates were {} and {} respectively.The guiding BLEU is BLEU-{guide}.\n'
    'The embeddings file used was {embedding}.The transforms used are: {transforms_applied}.\n'.format(run_number,dataset,fine_tune,
                                                                               best_bleu1,best_bleu2,best_bleu3,best_bleu4,
                                                                               epochs,epoch+1,emb_dim,attention_dim,
                                                                               decoder_dim,dropout,encoder_lrate,decoder_lr,
                                                                               guide=guiding_bleu, 
                                                                               embedding=emb_file_name if embeddings_file is not None else 'NA',
                                                                               transforms_applied=transforms_to_apply if transforms_to_apply != [] else 'Nil'))   
    if not fine_tune_encoder:
        increase_run_number()

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, image_name) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

   
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

 
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps, image_name) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data


            # Calculate loss
            loss = criterion(scores, targets)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        bleu3 = corpus_bleu(references, hypotheses, (1.0/3.0,1.0/3.0,1.0/3.0,))
        bleu2 = corpus_bleu(references, hypotheses, (1.0/2.0,1.0/2.0,))
        bleu1 = corpus_bleu(references, hypotheses, (1.0/1.0,))

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU scores - {bleu1},{bleu2},{bleu3},{bleu4}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu1=bleu1,bleu2=bleu2,bleu3=bleu3,bleu4=bleu4))

    return bleu1,bleu2,bleu3,bleu4


if __name__ == '__main__':
    main()
