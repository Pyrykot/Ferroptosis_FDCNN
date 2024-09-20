import albumentations as A
import cv2 as cv
import os
import glob as glob
import config
import json
import openpyxl as exl
import openpyxl.utils as exl_utils
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from openpyxl.styles import Font
from natsort import natsorted
from config import CLASSES




def get_train_transform():
    return A.Compose([
        A.CLAHE(clip_limit=0.8, p=1.0, tile_grid_size=(8, 8)),
        A.RandomRotate90(p=0.5),
        A.ToFloat(always_apply=True),
        ToTensorV2(p=1.0)
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels'],
        'min_visibility': 0.5
    })


def get_test_transform():
    return A.Compose([
        A.CLAHE(clip_limit=0.8, p=1.0, tile_grid_size=(5, 5)),
        A.ToFloat(always_apply=True),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_inference_transform():
    return A.Compose([
        A.CLAHE(clip_limit=2, p=1.0, tile_grid_size=(5, 5)),
        A.ToFloat(always_apply=True),
    ])


def show_img_w_boxes(img, img_name="Image", boxes=None, labels=None, scale_factor=3):
    """"
    Handy funciton to show images with bboxes

    :param img: Image to be shown
    :param img_name: Optional image name
    :param boxes: Optional bounding boxes
    :param labels: Optional labels, if labels exist boxes with 1 = green, everything else yellow
    :param scale_factor: Upscale image, default = 3
    """
    img = np.transpose(img, (1, 2, 0)).copy()
    width = img.shape[1] * scale_factor
    heigth = img.shape[0] * scale_factor
    img = cv.resize(img, (width, heigth), interpolation=cv.INTER_AREA)
    if boxes is not None:
        for j, box in enumerate(boxes):
            x_min = int(box[0]) * scale_factor
            y_min = int(box[1]) * scale_factor
            x_max = int(box[2]) * scale_factor
            y_max = int(box[3]) * scale_factor
            if labels is not None:
                if labels[j] == 1:
                    cv.rectangle(img, (x_min - 1, y_min - 1), (x_max + 1, y_max + 1), (0, 255, 0), 1)
                else:
                    cv.rectangle(img, (x_min - 1, y_min - 1), (x_max + 1, y_max + 1), (0, 255, 255), 1)

    winname = os.path.basename(img_name)
    cv.namedWindow(winname)
    cv.moveWindow(winname, 500, 300)
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_gt_boxes(path, min_size=8):
    """"
    Loads ground truth boxes from geoJSON file.

    :param path: path to the file
    :param min_size: Minimum size of width or height of box.
    """
    boxes = []
    labels = []
    with open(path, 'r') as f:
        data = json.load(f)
        data = data['features']

    for y in data:
        xmin = int(y['geometry']['coordinates'][0][0][0])
        xmax = int(y['geometry']['coordinates'][0][1][0])
        ymin = int(y['geometry']['coordinates'][0][1][1])
        ymax = int(y['geometry']['coordinates'][0][3][1])

        if (xmax - xmin) > min_size and (ymax - ymin) > min_size:
            labels.append(CLASSES.index(y['properties']['classification']['name']))
            boxes.append([xmin, ymin, xmax, ymax])
    return boxes, labels, data


def load_data_keyword(img_path, keywords, transforms):
    """"
    Load images and labels using keywords
    """
    imgs, img_names, gt_boxes, gt_labels = ([] for i in range(4))
    image_paths = natsorted(glob.glob(img_path + "/*.png"))

    for i in range(len(image_paths)):
        image_name = image_paths[i].split('/')[-1].split('.')[0]
        if any(ext.lower() in image_name.lower() for ext in keywords):
            image = cv.imread(image_paths[i])
            if transforms is not None:
                image = transforms(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(float)
            imgs.append(image)
            img_names.append(image_name)
            json_name = os.path.basename(image_name) + ".geojson"

            boxes, labels, _ = get_gt_boxes(path=os.path.join(img_path, json_name))
            keep = cleanup(boxes, img_dim=image.shape, edge_limit=5)
            gt_boxes.append([boxes[idx] for idx in keep])
            gt_labels.append([labels[idx] for idx in keep])

    return imgs, img_names, gt_boxes, gt_labels


def cleanup(boxes, edge_limit=10, img_dim=(3, 100, 100), size_limit=5):
    keep = []
    for j, bbox in enumerate(boxes):
        min_x = bbox[0]
        min_y = bbox[1]
        max_x = bbox[2]
        max_y = bbox[3]

        width = max_x - min_x
        height = max_y - min_y
        if min_x > edge_limit and min_y > edge_limit and max_x < (img_dim[2] - edge_limit) and max_y < (
                img_dim[1] - edge_limit) and width > size_limit and height > size_limit:
            keep.append(j)

    return keep

def align_gt_to_pred_scores(ground_truth_boxes, predicted_boxes, pr_scores, iou_threshold, pred_labels):
    pred_scores = []
    no_prediction_box = []
    pred_label = []

    for gtBox in ground_truth_boxes:
        max_iou = 0
        index_of_max = 0
        for i, pr_box in enumerate(predicted_boxes):
            iou = IoU_calc(gtBox, pr_box)
            if iou > max_iou:
                max_iou = iou
                index_of_max = i

        if max_iou >= iou_threshold:
            pred_label.append(pred_labels[index_of_max])
            if pred_labels[index_of_max] == config.positive_label:
                pred_scores.append(pr_scores[index_of_max])  # the best prediction score will be set for that gt box
            else:
                pred_scores.append(1 - pr_scores[index_of_max])  # the best prediction score will be set for that gt box
        else:
            pred_scores.append(0.0)  # if gt does not have any good enough prediction the prob will be set to 0
            no_prediction_box.append(gtBox)
            pred_label.append(-1)

    return pred_scores, no_prediction_box, pred_label


def save_bboxes_to_json(bboxes, predictions, name):
    """
    :bboxes: List of bboxes coordinates [[minX, minY, maxX, maxY]..]
    :predictions: Predicted classes for each bbox
    :name: Filename for the geojson file
    """

    features = []
    for i, bbox in enumerate(bboxes):
        min_x = int(bbox[0])
        min_y = int(bbox[1])
        max_x = int(bbox[2])
        max_y = int(bbox[3])

        coordinates = [[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]]]
        geometry = {'type': 'Polygon', 'coordinates': coordinates}
        if predictions[i] == 2:
            color = [100, 54, 66]
        else:
            color = [163, 87, 219]
        classification = {'name': config.CLASSES[predictions[i]], 'color': color}
        properties = {'objectType': 'annotation', 'classification': classification}
        feature = {"type": "Feature", "id": "none", 'geometry': geometry, 'properties': properties}
        features.append(feature)

    output_dir = os.path.join(config.OUT_DIR, name.split("_")[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = {'type': 'FeatureCollection', 'features': features}
    if len(data['features']) > 0:
        with open(os.path.join(output_dir, f'{os.path.basename(name)}.geojson'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)


def print_to_excel_inference(platemap, name):
    workbook = exl.Workbook()
    sheet = workbook.active
    unique_timestamps = platemap.get_unique_timestamps()
    # Write title
    sheet['A1'] = f'Vessel name:'
    sheet['B1'] = name
    sheet['A2'] = "Normal"
    sheet['A3'] = 'Elapsed'
    bold_font = Font(bold=True)
    ferr_start_row = len(unique_timestamps) + 5
    sheet[f'A{ferr_start_row}'] = 'Ferroptosis'
    sheet[f'A{ferr_start_row + 1}'] = 'Elapsed'

    # Write timestamps
    for i, timestamp in enumerate(unique_timestamps):
        sheet[f'A{i + 4}'] = timestamp
        sheet[f'A{i + ferr_start_row + 2}'] = timestamp
        sheet[f'A{i + ferr_start_row + 2}'].font = bold_font
        sheet[f'A{i + 4}'].font = bold_font

    col_index = 0
    for well in platemap:
        norm_stamps, ferr_stamps, empty = platemap.get_results_iter()
        if empty:
            continue

        col_index += 1
        col = exl_utils.get_column_letter(col_index + 1)
        condition = platemap.get_condition_string_iter()
        sheet[f'{col}{3}'] = condition
        sheet[f'{col}{3}'].font = bold_font

        sheet[f'{col}{ferr_start_row + 1}'] = condition
        sheet[f'{col}{ferr_start_row + 1}'].font = bold_font

        for j, norm in enumerate(norm_stamps):
            row = j + 4  # timestamps.index(well['timestamps'][j])
            sheet[f'{col}{row}'] = norm

        for j, ferr in enumerate(ferr_stamps):
            row = j + ferr_start_row + 2
            sheet[f'{col}{row}'] = ferr

    # Save the workbook to a file
    savepath = f"{config.OUT_DIR}/{name}.xlsx"
    workbook.save(savepath)

    # Close the workbook
    workbook.close()

def preload_well_phase(well):
    imgs = []
    times = []
    inds = []
    paths = []
    for time, timestamp_list in enumerate(well['image_paths']):
        for ind, path in enumerate(timestamp_list):
            if path is not None:
                image = cv.imread(path)
                image = get_inference_transform()(image=image)['image']
                image = np.transpose(image, (2, 0, 1)).astype(float)

                imgs.append(image)
                paths.append(path)
                times.append(time)
                inds.append(ind)
    return imgs, times, inds, paths

def plots(y_true, scores, model, print_prrec=False, threshold=0.8, name="Plots"):
    scores_thresholded = [1 if prob >= 0.8 else 0 for prob in scores[0]]
    binary_clasification = [item - 1 for item in y_true]
    F1 = metrics.f1_score(binary_clasification, scores_thresholded)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
    fig.canvas.manager.set_window_title(name)
    auc = 0
    precision, recall, thresholds = (0, 0, 0)
    for yscore in scores:
        fpr, tpr, thresholds_roc = metrics.roc_curve(y_true, yscore, pos_label=config.positive_label)
        auc = metrics.roc_auc_score(y_true, yscore)

        precision, recall, thresholds = metrics.precision_recall_curve(y_true, yscore, pos_label=config.positive_label)
        axs[0].plot(recall, precision, color='purple', label=f'Model {model}')
        axs[0].set_ylabel('Precision')
        axs[0].set_xlabel('Recall')

        axs[1].plot(fpr, tpr, label=f"auc {model} =" + str(auc))
        axs[1].legend(loc=4)
        axs[1].set_ylabel('TPR')
        axs[1].set_xlabel('FPR')

        axs[2].plot(thresholds, precision[:-1], marker='.', label=f'Precision {model}')
        axs[2].plot(thresholds, recall[:-1], marker='.', label=f'Recall {model}')
        axs[2].set_xlabel('Cutoff Threshold')
        axs[2].set_ylabel('Precision')
        axs[2].legend(loc=4)

    #plt.show()
    if print_prrec:
        closest_index = np.argmin(np.abs(thresholds - threshold))
        print(f"{name} {threshold:.2f} {precision[closest_index]:.2f} {recall[closest_index]:.2f} {F1}{auc}")


def IoU_calc(box_a, box_b):
    A_min_x = box_a[0]
    A_min_y = box_a[1]
    A_max_x = box_a[2]
    A_max_y = box_a[3]

    B_min_x = box_b[0]
    B_min_y = box_b[1]
    B_max_x = box_b[2]
    B_max_y = box_b[3]

    intersection_area = max(0, min(A_max_x, B_max_x) - max(A_min_x, B_min_x)) * max(
        0, min(A_max_y, B_max_y) - max(A_min_y, B_min_y))

    A_area = (A_max_x - A_min_x) * (A_max_y - A_min_y)
    B_area = (B_max_x - B_min_x) * (B_max_y - B_min_y)
    tot_area = A_area + B_area - intersection_area
    IoU = intersection_area / tot_area
    return IoU


