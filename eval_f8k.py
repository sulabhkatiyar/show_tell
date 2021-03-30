import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import argparse


# Parameters
data_folder = '/home/sulabh02/for_jupyter/papers/iccv/pos/show_tell_vinyals/caption_data/'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'   # base name shared by data files
checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = '/home/sulabh02/for_jupyter/papers/iccv/pos/show_tell_vinyals/caption_data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

captions_dump=True


# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

choice=0
embeddings_ensemble_available=False

#Arguments to main()
parser = argparse.ArgumentParser(description = 'Evaluation of IC model')
parser.add_argument('beam_size', type=int,  help = 'Beam size for evaluation')
args = parser.parse_args()


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    #All global parameters here
    global captions_dump, data_name, choice, embeddings_ensemble_available
    
    empty_hypo = 0
    
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    captions_dict=dict()

    # Create a list of image names in same order as images
    image_names = list()

    # For each image
    for i, (image, caps, caplens, allcaps, image_name) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        if choice==0:
            # h, c = decoder.init_hidden_state(encoder_out)
            # h1, c1 = h.clone(), c.clone()
            # h2, c2 = h.clone(), c.clone()
            # h3, c3 = h.clone(), c.clone()
            # h, c, h1, c1, _, _, _, _ = decoder.init_hidden_state(encoder_out)     # Initialize LSTM state
            # h, c, h1, c1, h2, c2, _, _ = decoder.init_hidden_state(encoder_out)     # Initialize LSTM state
            # h, c, h1, c1, h2, c2, h3, c3 = decoder.init_hidden_state(encoder_out)     # Initialize LSTM state
            # h, c, h1, c1, h2, c2, h3, c3, h4, c4 = decoder.init_hidden_state(encoder_out)     # Initialize LSTM state
            h, c, h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = decoder.init_hidden_state(encoder_out)     # Initialize LSTM state


        elif choice==1:
            _,_,h_f,h_b,c_f,c_b = decoder.init_hidden_state_bidirectional(encoder_out)
            # h = h_f.add(torch.flip(h_b, (0,)))
            # c = c_f.add(torch.flip(c_b, (0,)))
        elif choice==2:
            h_1, h_2, h_3, c_1, c_2, c_3 = decoder.init_hidden_state_deeplstm(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            # adding features tensor
            # feature_map = decoder.features_tensor(encoder_out, k_prev_words).squeeze(1)

            if embeddings_ensemble_available == True:
                embeddings = decoder.final_embeddings(decoder.embedding(k_prev_words)).squeeze(1)  # (s, embed_dim)
            else:
                embeddings = decoder.embedding(k_prev_words).squeeze(1)
            # assert(feature_map.shape == embeddings.shape)
            # embeddings = embeddings + feature_map
            #awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            #gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            #awe = gate * awe
            if choice==0:
                h, c = decoder.decode_step(embeddings, (h, c))  # (s, decoder_dim)
                h1, c1 = decoder.decode_step1(h, (h1, c1))     # (batch_size_t, decoder_dim)
                h2, c2 = decoder.decode_step2(h1, (h2, c2))     # (batch_size_t, decoder_dim)
                h3, c3 = decoder.decode_step3(h2, (h3, c3))     # (batch_size_t, decoder_dim)
                h4, c4 = decoder.decode_step4(h3, (h4, c4))     # (batch_size_t, decoder_dim)
                h5, c5 = decoder.decode_step5(h4, (h5, c5))     # (batch_size_t, decoder_dim)

            elif choice==1:
                # print(embeddings.shape, h_f.shape)
                h_f, c_f = decoder.decode_step_f(embeddings, (h_f, c_f))
                # h_b, c_b = decoder.decode_step_b(torch.flip(embeddings, (0,1)), (h_b, c_b))
                # h = h_f.add(torch.flip(h_b, (0,)))
                # c = c_f.add(torch.flip(c_b, (0,)))
                # h = h_f.add(h_b)
                # c = c_f.add(c_b)
            elif choice == 2:
                h_1, c_1 = decoder.decode_step1(embeddings, (h_1, c_1))
                h_2, c_2 = decoder.decode_step2(h_1, (h_2, c_2))
                h_3, c_3 = decoder.decode_step3(h_2, (h_3, c_3))
                h, c = h_3, c_3
            # scores = decoder.fc(h)  # (s, vocab_size)
            # scores = decoder.fc(h1)  # (s, vocab_size)
            # scores = decoder.fc(h2)  # (s, vocab_size)
            # scores = decoder.fc(h3)  # (s, vocab_size)
            # scores = decoder.fc(h4)  # (s, vocab_size)
            scores = decoder.fc(h5)  # (s, vocab_size)


            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            if choice==0:
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                h1 = h1[prev_word_inds[incomplete_inds]]
                c1 = c1[prev_word_inds[incomplete_inds]]
                h2 = h2[prev_word_inds[incomplete_inds]]
                c2 = c2[prev_word_inds[incomplete_inds]]
                h3 = h3[prev_word_inds[incomplete_inds]]
                c3 = c3[prev_word_inds[incomplete_inds]]
                h4 = h4[prev_word_inds[incomplete_inds]]
                c4 = c4[prev_word_inds[incomplete_inds]]
                h5 = h5[prev_word_inds[incomplete_inds]]
                c5 = c5[prev_word_inds[incomplete_inds]]

            elif choice==1:
                h_f, c_f = h_f[prev_word_inds[incomplete_inds]], c_f[prev_word_inds[incomplete_inds]]
                # h_b, c_b = torch.flip(h_b[prev_word_inds[incomplete_inds]], (0,)), torch.flip(c_b[prev_word_inds[incomplete_inds]], (0,))
                # h_b, c_b = h_b[prev_word_inds[incomplete_inds]], c_b[prev_word_inds[incomplete_inds]]
            elif choice==2:
                h_1, c_1 = h_1[prev_word_inds[incomplete_inds]], c_1[prev_word_inds[incomplete_inds]]
                h_2, c_2 = h_2[prev_word_inds[incomplete_inds]], c_2[prev_word_inds[incomplete_inds]]
                h_3, c_3 = h_3[prev_word_inds[incomplete_inds]], c_3[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        try:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        except:
            seq = []
            empty_hypo += 1
            # continue

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        # Image names list
        image_names.append(image_name)

        assert len(references) == len(hypotheses) == len(image_names)

    # Print the number of hypotheses which remain empty        
    print('The number of empty hypotheses is {}'.format(empty_hypo))

    #save reference and hypotheses to a seperate file as well
    captions_dict['references']=references
    captions_dict['hypotheses']=hypotheses  
    captions_dict['image_names'] = image_names              
    #dump generated caption dict into .json file
    if captions_dump==True:
        with open('generated_captions_f8k.json', 'w') as gencap:
            json.dump(captions_dict, gencap)
        save_captions_mscoco_format(word_map_file,references,hypotheses,image_names,str(beam_size)+'_f8ktest')
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    bleu3 = corpus_bleu(references, hypotheses, (1.0/3.0,1.0/3.0,1.0/3.0,))
    bleu2 = corpus_bleu(references, hypotheses, (1.0/2.0,1.0/2.0,))
    bleu1 = corpus_bleu(references, hypotheses, (1.0/1.0,))
    return bleu1,bleu2,bleu3,bleu4


def main():
    beam_size = args.beam_size
    was_fine_tuned=False
    scores=evaluate(args.beam_size)
    print("\nBLEU scores @ beam size of %d is %.4f, %.4f, %.4f, %.4f." % (beam_size, scores[0],scores[1],scores[2],scores[3]))
    with open('eval_run_logs.txt', 'a') as eval_run:
        eval_run.write('The model is trained on {dataname} and {was} fine tuned.\n'
                       'The BLEU scores are {bleu_1}, {bleu_2}, {bleu_3}, {bleu_4}.\n'
                       'The beam_size was {beam}.The embedding was {embedding} and the \n'
                       'decoder was {decoder} and the dropout was {drop}.\n'
                       'The model was trained for {epochs} epochs.\n\n\n'.format(dataname=data_name,
                                                          was ='was' if was_fine_tuned==True else 'was not', 
                                                          bleu_1=scores[0], bleu_2=scores[1], bleu_3=scores[2],
                                                          bleu_4=scores[3], beam=beam_size, embedding=
                                                          decoder.embedding, decoder=[decoder.decode_step],
                                                          drop=decoder.dropout, epochs=checkpoint['epoch']))

if __name__ == '__main__':
    main()
