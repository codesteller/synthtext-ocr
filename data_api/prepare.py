import os
from pathlib import Path
from random import sample, seed
from tqdm import tqdm
import tensorflow as tf 
import cv2
import numpy as np
from scipy.io import loadmat
import json
from multiprocessing import Pool
from data_api.preprocess import make_heatmap 


synthtext_dir = "/home/codesteller/datasets/02_Business_Card/02_SynthText/SynthText/SynthText"
gt_dir = "/home/codesteller/datasets/02_Business_Card/02_SynthText/SynthText/OCR_Data"

class Dataset:
    def __init__(self, dbdir=synthtext_dir, out_dir=gt_dir, extn='jpg'):
        self.dbdir = dbdir
        self.extn = extn
        self.impaths = list()
        self.gtmat = os.path.join(self.dbdir, 'gt.mat')
        self.out_dir = out_dir
        self.do_parallel = True
        self.json_dir = None

    def parse_and_filter_mat_data(self, mat_data, file_filter):
        dataset = {}
        for im_file, wordBB, charBB, text in tqdm(zip(mat_data['imnames'][0], mat_data['wordBB'][0], mat_data['charBB'][0],
                                                    mat_data['txt'][0]), desc='Parsing & filtering gt.mat'):
            if im_file[0] in file_filter:
                dataset[im_file[0]] = {'wordBB': wordBB, 'charBB': charBB, 'text': text}
        return dataset

    def extract_dataset(self, samples=True):
        mat_file = Path(self.gtmat)
        img_root = mat_file.parent
        out_dir = Path(self.out_dir)
        self.json_dir = out_dir / 'ocr_gt'

        # files in dataset folder
        imgs = set([img.relative_to(img_root).as_posix() for img in tqdm(img_root.glob('**/*.jpg'))])

        # load dataset
        mat_data = loadmat(mat_file)
        print('{} loaded.'.format(mat_file.as_posix()))

        # filter the gt_mat data to only display the data from images
        gt_dataset = self.parse_and_filter_mat_data(mat_data, imgs)
        print('Parsing complete')

        json_label = Path(os.path.join(self.json_dir, "ocr_{}_gt.json".format(len(gt_dataset))))

        # Check if already exists, the load and return else extract dataset
        if os.path.exists(json_label):
            resp = input("GT already available. DO you want to proceed (y/N): ")
            if resp.lower() != 'y':
                with open(json_label, "r") as fptr:
                   ocr_dataset = json.load(fptr) 
                print('{} loaded.'.format(json_label.as_posix()))
                return gt_dataset, ocr_dataset
            else:
                print("Starting Dataset Extraction")

        # print samples, atleast 1% of generated data or 1
        n_samples = 10
        print('Generating image samples for {}'.format(n_samples))
        self.print_sample(gt_dataset, img_root, out_dir / 'samples', n_samples=n_samples)

        # Generate OCR Data in JSON format
        print('Generating Training Dataset')
        training_samples = 10
        ocr_dataset = self.generate_ocr_json(gt_dataset, img_root, self.json_dir, training_samples)
        

        return gt_dataset, ocr_dataset

    def generate_ocr_json(self, gt_dataset, data_dir, json_dir, training_samples=1):
        # just make sure that output dir is writable
        # and we have permissions and stuff
        db = dict()
        json_dir.mkdir(parents=True, exist_ok=True)
        sample_i = 0

        if self.do_parallel:
            pool = Pool(os.cpu_count())           # Create a multiprocessing Pool
            data_outputs = pool.map(self._write_ocr_data_parallel, tqdm(gt_dataset.items()))  # process data_inputs iterable with pool
            pool.close()
            pool.join()

            # Dump Data to Disk
            print("Writing GT to Disk Started....")
            out_file = Path(os.path.join(json_dir, "ocr_{}_gt.json".format(len(gt_dataset))))
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, "w") as fptr:
                json.dump(data_outputs, fptr)

            print("Writing GT to Disk Complete....")

            return data_outputs

        else:
            for im_file, gt_data in tqdm(gt_dataset.items(), desc='Creating Sample files [{}]'.format(training_samples)):
                if sample_i >= training_samples:
                    break
                sample_i += 1 

                db[im_file] = self._write_ocr_data(data_dir, gt_data, im_file, json_dir)
            return db

    @staticmethod
    def _write_ocr_data_parallel(gt_datum, write_individual=False):
        im_file = gt_datum[0]
        gt_data = gt_datum[1]
        data_dir = synthtext_dir
        samples_dir = os.path.join(gt_dir, 'ocr_gt')

        all_words = []
        gt_json = dict()
        for j in gt_data['text']:
            all_words += [k for k in ' '.join(j.split('\n')).split() if k != '']
        n_words = len(all_words)

        # BB word boxes and word text on top of box
        gt_json["image_path"] = os.path.join(data_dir, im_file)
        gt_json["num_words"] = n_words
        gt_json["word_list"] = all_words

        gt_json["labels"] = list()
        for word_i in range(n_words):
            temp = dict()
            word_bbox = gt_data['wordBB']
            if 2 == len(word_bbox.shape):
                word_bbox = gt_data['wordBB'][:, :, np.newaxis]
            word_pts = word_bbox[:, :, word_i].transpose().reshape((-1, 1, 2)).astype(np.int)

            # Fill the dictionary
            temp["word"] = all_words[word_i]
            temp["bbox"] = word_pts.tolist()
            gt_json["labels"].append(temp)

        if write_individual:
            # Dump the data in disk
            out_file = Path(os.path.join(samples_dir, (os.path.splitext(im_file)[0] + '_gt.json')))
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, "w") as fptr:
                json.dump(gt_json, fptr)


        return gt_json

    @staticmethod
    def _write_ocr_data(data_dir, gt_data, im_file, sample_dir):
        all_words = []
        gt_json = dict()
        for j in gt_data['text']:
            all_words += [k for k in ' '.join(j.split('\n')).split() if k != '']
        n_words = len(all_words)

        # BB word boxes and word text on top of box
        gt_json["image_path"] = (data_dir / im_file).as_posix()
        gt_json["num_words"] = n_words
        gt_json["word_list"] = all_words

        gt_json["labels"] = list()
        for word_i in range(n_words):
            temp = dict()
            word_bbox = gt_data['wordBB']
            if 2 == len(word_bbox.shape):
                word_bbox = gt_data['wordBB'][:, :, np.newaxis]
            word_pts = word_bbox[:, :, word_i].transpose().reshape((-1, 1, 2)).astype(np.int)

            # Fill the dictionary
            temp["word"] = all_words[word_i]
            temp["bbox"] = word_pts.tolist()
            gt_json["labels"].append(temp)
            
        # gt_json["labels"] = temp   

        # Dump the data in disk
        out_file = sample_dir / (os.path.splitext(im_file)[0] + '_gt.json')
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w") as fptr:
            json.dump(gt_json, fptr)

        return gt_json
            

    def print_sample(self, gt_dataset, data_dir, sample_dir, n_samples=1):
        if n_samples <= 0:
            return

        # just make sure that output dir is writable
        # and we have permissions and stuff
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_i = 0
        for im_file, gt_data in tqdm(gt_dataset.items(), desc=f'Creating Sample files [{n_samples}]'):
            if sample_i >= n_samples:
                break

            self._write_image_sample(data_dir, gt_data, im_file, sample_dir)
            self._write_region_maps_sample(data_dir, gt_data, im_file, sample_dir)
            sample_i += 1

    @staticmethod
    def _write_image_sample(data_dir, gt_data, im_file, sample_dir):
        im = cv2.imread((data_dir / im_file).as_posix())
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        all_words = []
        for j in gt_data['text']:
            all_words += [k for k in ' '.join(j.split('\n')).split() if k != '']
        n_words = len(all_words)
        # draw word boxes and word text on top of box
        for word_i in range(n_words):
            word_bbox = gt_data['wordBB']
            if 2 == len(word_bbox.shape):
                word_bbox = gt_data['wordBB'][:, :, np.newaxis]
            word_pts = word_bbox[:, :, word_i].transpose().reshape((-1, 1, 2)).astype(np.int)
            word_pts1 = np.array([[420, 21], [512, 23], [511, 41], [420, 39]])
            word_pts1 = word_pts1.reshape((-1, 1, 2))
            cv2.polylines(im, word_pts1, True, color=(255, 0, 0), thickness=2)
            cv2.putText(im, all_words[word_i], tuple(word_pts[0, 0].tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), lineType=cv2.LINE_AA)
        # draw the character boxes and char associated with each box
        all_chars = []
        for word_i in range(n_words):
            all_chars += all_words[word_i]
        for char_i in range(len(all_chars)):
            char_pts = gt_data['charBB'][:, :, char_i].transpose().reshape((-1, 1, 2)).astype(np.int)
            cv2.polylines(im, char_pts, True, (0, 255, 255))
            cv2.putText(im, all_chars[char_i], tuple(char_pts[-1, 0].tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), lineType=cv2.LINE_AA)
        out_file = sample_dir / (os.path.splitext(im_file)[0] + '_input.png')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_file.as_posix(), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _write_region_maps_sample(data_dir, gt_data, im_file, sample_dir):
        char_bbox = gt_data['charBB']
        text = gt_data['text']

        rg_map, aff_map, _ = make_heatmap(data_dir / im_file, char_bbox, text)
        cv2.imwrite('test_aff_map.png', cv2.applyColorMap(aff_map, cv2.COLORMAP_JET))

        # write the character map
        out_file = sample_dir / (os.path.splitext(im_file)[0] + '_char_rg.png')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_file.as_posix(), cv2.applyColorMap(rg_map, cv2.COLORMAP_JET))

        # write the affinity map
        out_file = sample_dir / (os.path.splitext(im_file)[0] + '_affinity_rg.png')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_file.as_posix(), cv2.applyColorMap(aff_map, cv2.COLORMAP_JET))



def main():
    db = Dataset(synthtext_dir)
    # db.get_imagepaths()

    # print(len(db.impaths))

    gt_dataset, ocr_dataset = db.extract_dataset()    


if __name__ == "__main__":
    main()   

