from data_api.prepare import Dataset
import json


def main():
    # synthtext_dir = "/home/codesteller/datasets/02_Business_Card/02_SynthText/SynthText/SynthText"
    # gt_dir = "/home/codesteller/datasets/02_Business_Card/02_SynthText/SynthText/OCR_Data"
    # db = Dataset(synthtext_dir, out_dir=gt_dir)
    # # db.get_imagepaths()

    # # print(len(db.impaths))

    # _, ocr_dataset = db.extract_dataset()    

    json_file = "/home/codesteller/datasets/02_Business_Card/02_SynthText/SynthText/OCR_Data/ocr_gt/ocr_858750_gt.json"
    with open(json_file, 'r') as fptr:
        ocr_dataset = json.load(fptr)

    print("Samples: ", len(ocr_dataset))

    for isample in ocr_dataset[:10]:
        image_path = isample["image_path"]
        word_list = isample["word_list"]

        print(image_path, word_list)




if __name__ == "__main__":
    main()  