import datasets
import Inference
import torch
import matplotlib.pyplot as plt
import os
import config
import utils as utils
import sklearn.metrics as metrics
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from model import create_model
from tqdm import tqdm
from torchvision.ops import boxes as box_ops
plt.style.use('ggplot')


def main():
    """
    Constructs the neccessary objects for training loop
    """

    # Loads images for cross validation testing. FasterRCNN_dataset implementation pending
    images, image_names, gt_boxes, gt_labels = utils.load_data_keyword(config.TESTING_DIR, ["Kar", "Oci", "Pf", "Ri"],
                                                                       utils.get_inference_transform())

    train_loader = datasets.get_FCNN_trainloader(utils.get_train_transform(),
                                                 keywords=["Kar", "Oci", "Pf", "Ri"])
    valid_loader = datasets.get_FCNN_testloader(utils.get_test_transform(),
                                                keywords=["Kar", "Oci", "Pf", "Ri"])
    model = training_loop(train_loader, valid_loader, config.LEARNING_RATE, config.WEIGHT_DECAY).half().eval()
    f1, precision, recall = test_for_metrics(model, images, image_names, gt_boxes, gt_labels)
    print(f"F1 {f1}, precision {precision}, recall {recall}")


def training_loop(train_loader, valid_loader, lr, decay):
    """
    The training loop. Optimizer and scheduler are constucted. Training is continued until epoch limit is reached.
    """
    global optimizer, scheduler


    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=lr, momentum=config.OPTIM_MOMENTUM, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.MILESTONES, gamma=config.GAMMA)

    train_loss_list = [] # Stores the training loss
    val_loss_list = [] # Stores the validation loss
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.eval()
        model.to(torch.float32).train()
        tot_train_loss, iters = train(train_loader, model)
        train_loss_list.append(tot_train_loss / iters)
        tot_val_loss, val_itr = validate(valid_loader, model)
        val_loss_list.append(tot_val_loss / val_itr)
        scheduler.step()

        if epoch == config.SAVE_MODEL_EPOCH:
            torch.save(model.state_dict(), f"{OUT_DIR}/{config.MODEL_NAME}")

        if (epoch + 1) % config.SAVE_PLOTS_EPOCH == 0:
            plt.figure()

            plt.plot(val_loss_list, label='Validation loss', marker='o')
            plt.plot(train_loss_list, label='Training loss', marker='x')
            plt.legend()
            plt.title('Losses')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig(f"{OUT_DIR}/{config.MODEL_NAME}_loss.png")
            plt.close()

    torch.cuda.empty_cache()
    return model

def train(train_data_loader, model):
    """
    Iteration loop. Loops trough train data and passes it to the model
    """
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    iters = len(train_data_loader)
    total_loss = 0.0
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        total_loss += loss_value
        losses.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    return total_loss, iters


def validate(valid_data_loader, model):
    """
    For cross validation loss.
    """
    iters = len(valid_data_loader)
    tot_loss = 0.0
    for i, data in enumerate(valid_data_loader):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        tot_loss += loss_value
        torch.cuda.empty_cache()
    return tot_loss, iters


def test_for_metrics(model, images, image_names, gt_boxes, gt_labels):
    """
    Test model performance with scikit metrics library
    """
    with torch.inference_mode():
        aligned_scores = []
        for image, image_name, gt_boxes, gt_labels in zip(images, image_names, gt_boxes, gt_labels):
            boxes, labels, scores = Inference.batched_inference(image, config.CHUNK_SIZE, model)
            labellss = torch.ones(len(labels), device=DEVICE)
            keep_nms = box_ops.batched_nms(boxes, scores, labellss, 0.2)

            boxes = boxes.detach().cpu().data.tolist()
            scores = scores.detach().cpu().data.tolist()
            labels = labels.detach().cpu().data.tolist()

            boxes = [boxes[idx] for idx in keep_nms]
            labels = [labels[idx] for idx in keep_nms]
            scores = [scores[idx] for idx in keep_nms]
            iou_treshold = 0.5
            aligned, _, _ = utils.align_gt_to_pred_scores(ground_truth_boxes=gt_boxes, predicted_boxes=boxes,
                                                          iou_threshold=iou_treshold, pr_scores=scores,
                                                          pred_labels=labels)
            aligned_scores.extend(aligned)
    torch.cuda.empty_cache()
    y_true = [item - 1 for sublist in gt_labels for item in sublist]
    predictions = [1 if prob >= 0.8 else 0 for prob in aligned_scores]
    f1 = metrics.f1_score(y_true, predictions)
    precision = metrics.precision_score(y_true, predictions)
    recall = metrics.recall_score(y_true, predictions)
    return f1, precision, recall


if __name__ == '__main__':
    main()
