import xml.etree.ElementTree as ET
import os

from tqdm import tqdm


CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )


def parse_voc_annotation(
    data_path, file_type, anno_path, use_difficult_bbox=False
):
    """
    phase pascal voc annotation, eg:[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: eg: VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param file_type: eg: 'trainval''train''val'
    :param anno_path: path to ann file
    :param use_difficult_bbox: whither use different sample
    :return: batch size of data set
    """
    img_inds_file = os.path.join(
        data_path, "ImageSets", "Main", file_type + ".txt"
    )
    with open(img_inds_file, "r") as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open(anno_path, "a") as f:
        for image_id in tqdm(image_ids):
            image_path = os.path.join(
                data_path, "JEPGImages", image_id + ".jpg"
            )
            annotation = image_path
            label_path = os.path.join(
                data_path, "Annotations", image_id + ".xml"
            )
            root = ET.parse(label_path).getroot()
            objects = root.findall("object")
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (
                    int(difficult) == 1
                ):
                    continue
                bbox = obj.find("bndbox")
                class_id = CLASSES.index(obj.find("name").text.lower().strip())
                xmin = bbox.find("xmin").text.strip()
                ymin = bbox.find("ymin").text.strip()
                xmax = bbox.find("xmax").text.strip()
                ymax = bbox.find("ymax").text.strip()
                annotation += " " + ",".join(
                    [xmin, ymin, xmax, ymax, str(class_id)]
                )
            annotation += "\n"
            f.write(annotation)
    return len(image_ids)


if __name__ == "__main__":
    # train_set :  VOC2007_trainval 和 VOC2012_trainval
    train_data_path_2007 = os.path.join(
        "/data2/wsj", "VOCtrainval-2007", "VOCdevkit", "VOC2007"
    )
    train_data_path_2012 = os.path.join(
        "/data2/wsj", "VOCtrainval-2012", "VOCdevkit", "VOC2012"
    )
    train_annotation_path = os.path.join("/data2/wsj", "train_annotation.txt")
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)

    # val_set   : VOC2007_test
    test_data_path_2007 = os.path.join(
        "/data2/wsj", "VOCtest-2007", "VOCdevkit", "VOC2007"
    )
    test_annotation_path = os.path.join("/data2/wsj", "test_annotation.txt")
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)

    len_train = parse_voc_annotation(
        train_data_path_2007,
        "trainval",
        train_annotation_path,
        use_difficult_bbox=False,
    )
    + parse_voc_annotation(
        train_data_path_2012,
        "trainval",
        train_annotation_path,
        use_difficult_bbox=False,
    )
    len_test = parse_voc_annotation(
        test_data_path_2007,
        "test",
        test_annotation_path,
        use_difficult_bbox=False,
    )

    print(
        "The number of images for train and test are :train : {0} | test : {1}".format(
            len_train, len_test
        )
    )