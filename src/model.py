import torchvision
import torch
import torch.nn.functional as F
import types
import config
from typing import List, Tuple
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch import Tensor
from torchvision.ops import boxes as box_ops


def create_model(num_classes):
    """
    Image detection model is constructed here.
    :num_classes: Number of classes in the dataset
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT',
                                                                    trainable_backbone_layers=config.TRAIN_LAYERS,
                                                                    anchor_generator=AnchorGenerator(
                                                                        sizes=config.ANCHOR_SIZES,
                                                                        aspect_ratios=config.ASPECT_RATIOS),
                                                                    box_detections_per_img=300,
                                                                    rpn_batch_size_per_image=512,
                                                                    max_size=config.UPSCALE,
                                                                    image_mean=config.IMG_MEAN,
                                                                    image_std=config.IMG_STD,
                                                                    rpn_pre_nms_top_n_train=4000,
                                                                    rpn_pre_nms_top_n_test=2000,
                                                                    rpn_post_nms_top_n_train=4000,
                                                                    rpn_post_nms_top_n_test=2000,
                                                                    rpn_score_thresh=0.6,
                                                                    rpn_fg_iou_thresh=0.5,
                                                                    rpn_bg_iou_thresh=0.3,
                                                                    box_fg_iou_thresh=0.6,
                                                                    box_bg_iou_thresh=0.3,
                                                                    )
    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # overwrite function to get raw logits
    model.roi_heads.postprocess_detections = types.MethodType(postprocess_detections, model.roi_heads)
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
    """
    The pytorch's postprocess detections contained NMS. NMS have been replaced after batched inference has been done to get accurate predictions for overlap areas.
    """
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)
        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        # remove low scoring boxes
        inds = torch.where(scores > 0.85)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


if __name__ == '__main__':
    print("")
