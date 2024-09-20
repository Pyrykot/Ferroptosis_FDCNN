import torch
import os
import config
import utils as utils
import math
import sys
from tqdm import tqdm
from platemap import PlateMap
from model import create_model
from config import OUT_DIR, DEVICE
from torchvision.ops import boxes as box_ops

def main(folder, model_name, save_boxes):
    """
    Loads neccessary objects for inference
    :param folder: name of the folder that contains images to be analyzed. This must be in folder defined in config as FILES_DIR
    :param model_name: Name of the model to be used. Must be located in folder defined in config as OUT_DIR.
    :param save_boxes: 1 = bboxes are saved as geojson format, 0 = boxes will not be saved
    """
    path_to_folder = os.path.join(config.FILES_DIR, folder)
    platemap = PlateMap(path_to_folder)
    with torch.inference_mode():
        model = create_model(num_classes=3).to(DEVICE).half()
        model.load_state_dict(torch.load(os.path.join(OUT_DIR, model_name), map_location=DEVICE))
        model.eval()
        analyze_data(model, platemap, save_bboxes=save_boxes)
    utils.print_to_excel_inference(platemap, folder)


def analyze_data(model, platemap, save_bboxes=0):
    """
    Main loop for inference. Loops trough platemap object.
    :param model: faster-RCNN model in inference mode
    :param platemap: Platemap class object, images to be infered are stored in the platemap object, raw counts will be saved in the platemap object
    :param save_bboxes: 1 = bboxes are saved as geojson format, 0 = boxes will not be saved. Default = 0
    """
    for well in tqdm(platemap):
        if well["empty_well"]:
            continue
        imgs, time_inds, inds, paths = utils.preload_well_phase(well) #Preloads every image in one well before continuing to inference.

        for image, time_ind, ind, path in zip(imgs, time_inds, inds, paths):
            with torch.no_grad():
                boxes, pred_classes, scores = batched_inference(image, config.CHUNK_SIZE, model)
            labels = torch.ones(len(pred_classes), device=DEVICE)
            keep_nms = box_ops.batched_nms(boxes, scores, labels, config.IOU_TRESHOLD_inference)
            boxes = boxes.detach().cpu().data.tolist()
            pred_classes = pred_classes.detach().cpu().data.tolist()

            boxes = [boxes[j] for j in keep_nms]
            pred_classes = [int(pred_classes[j]) for j in keep_nms]
            utils.show_img_w_boxes(image, "name", boxes, pred_classes, 1)

            if save_bboxes:
                utils.save_bboxes_to_json(boxes, pred_classes, os.path.basename(path)[:-4])

            # Insert results to platemap object. pred_classes contains as first entry "background" which is ignored.
            platemap.insert_results(time_ind, ind, (pred_classes.count(1), pred_classes.count(2)))

    # Save raw calls to xlsx file
    utils.print_to_excel_inference(platemap, config.excel_name)


def batched_inference(image, chunk_size, model):
    """
    Splits the image to chunks and feeds it to the model
    :image: Image file as a ndarray
    :chunk_size: Size of the chunck to be processed. Chunck size must be same as during training
    :model: Faster-rcnn model
    :return: bboxes, predicted labels and precition scores
    """
    channels, height, width = image.shape

    num_chunks_x = math.ceil((width / chunk_size))
    num_chunks_y = math.ceil((height / chunk_size))

    overlap_X = int(((num_chunks_x - 1) * chunk_size - (width - chunk_size)) / (num_chunks_x - 1))
    overlap_Y = int(((num_chunks_y - 1) * chunk_size - (height - chunk_size)) / (num_chunks_y - 1))
    all_boxes = torch.empty(0, device=config.DEVICE)
    all_labels = torch.empty(0, device=config.DEVICE)
    all_scores = torch.empty(0, device=config.DEVICE)
    tensor_chunks = []

    move_img_transforms = []  # List of starx, starty to move boxes to correct places

    for j, y in enumerate(range(0, height, chunk_size)):
        for i, x in enumerate(range(0, width, chunk_size)):
            if x == 0:
                startx = x
                endx = startx + chunk_size
            else:
                startx = x - (overlap_X * i)
                endx = startx + chunk_size

            if y == 0:
                starty = 0
                endy = starty + chunk_size
            else:
                starty = y - (overlap_Y * j)
                endy = starty + chunk_size
            tensor_chunks.append(
                torch.tensor(image[:, starty:endy, startx:endx], dtype=torch.half, device=config.DEVICE,
                             requires_grad=False))
            move_img_transforms.append((startx, starty))
    batch_size = 5
    all_outputs = []
    for i in range(0, len(tensor_chunks), batch_size):
        batch = tensor_chunks[i:i + batch_size]
        all_outputs.extend(model(batch))

    torch.cuda.empty_cache()
    for i, out_dict in enumerate(all_outputs):
        if len(out_dict['boxes']) != 0:
            startx = move_img_transforms[i][0]
            starty = move_img_transforms[i][1]
            all_boxes = torch.cat((all_boxes, out_dict['boxes'] + torch.tensor([startx, starty, startx, starty],
                                                                               device=config.DEVICE)))
            all_scores = torch.cat((all_scores, out_dict['scores']))
            all_labels = torch.cat((all_labels, out_dict['labels']))
    torch.cuda.empty_cache()
    return all_boxes, all_labels, all_scores





if __name__ == '__main__':
    # Sysargs
    # 1 Name of the folder with files
    # 2 model name in outputs folder
    # 3 save boxes 0 false, 1 true
    folder = sys.argv[1]
    model_name = sys.argv[2]
    save_boxes = int(sys.argv[3])
    main(folder, model_name, save_boxes)

