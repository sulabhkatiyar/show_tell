from utils import create_input_files

"""
To create files that contain all images stored in h5py format and captions stored in json files.
Minimum word frequencies to be used as cut-off for removing rare words to be specifiied here.
"""

if __name__ == '__main__':
    create_input_files(dataset='coco',
                       karpathy_json_path='path_to___dataset_coco.json',
                       image_folder='path_to__mscoco_folder',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='folder_for_processed_data',
                       max_len=40)
