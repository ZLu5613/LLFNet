import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR
from models.lobe_fewshot import Lobe_fewShotSeg
from dataloading.datasets import LobeTrainDataset
from utils import *
from config import ex 


def train(train_loader, model, criterion, optimizer, scheduler, G_list, slicer_len, record, record_matrix):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    q_loss = AverageMeter('Query loss', ':.4f')
    a_loss = AverageMeter('Align loss', ':.4f')


    def log_clamped_probs(pred):
        """Apply log and clamp to predictions."""
        return torch.log(torch.clamp(pred, torch.finfo(torch.float32).eps, 1 - torch.finfo(torch.float32).eps))

    def update_G_list():
        """Update G_list with background and foreground probabilities."""
        G_list[0][0][slicer_idx], G_list[1][0][slicer_idx] = BGpro_n[0], FGpro_n[0]
        G_list[0][slicer_label][slicer_idx], G_list[1][slicer_label][slicer_idx] = BGpro_n[1], FGpro_n[1]
        record_matrix[slicer_label-1, slicer_idx] = 1

    # Train mode.
    model.train()

    end = time.time()
    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Extract episode data.
        support_images = [[shot.float().cuda() for shot in way] for way in sample['support_images']]
        support_fg_mask = [[shot.float().cuda() for shot in way] for way in sample['support_fg_labels']]
        support_bg_mask = [[shot.float().cuda() for shot in way] for way in sample['support_bg_labels']]
        query_images = [query_image.float().cuda() for query_image in sample['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

        slicer_idx = int(sample['slicer_idx'].item() * slicer_len)
        slicer_label = int(sample['slicer_label'].item())

        # Log loading time.
        data_time.update(time.time() - end)

        # Compute outputs and losses.
        query_pred, align_loss, BGpro_n, FGpro_n = model(support_images, support_fg_mask, support_bg_mask, query_images)

        query_loss = sum(criterion(log_clamped_probs(pred), query_labels[n]) for n, pred in enumerate(query_pred))
        loss = query_loss + align_loss

        if record:
            BGpro_n = [t.detach().cpu().numpy() for t in BGpro_n]
            FGpro_n = [t.detach().cpu().numpy() for t in FGpro_n]
            update_G_list()

        # Clear gradients, backward, and optimize
        optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss.
        losses.update(loss.item())
        q_loss.update(query_loss.item())
        a_loss.update(align_loss.item())


    # Log elapsed time.
    batch_time.update(time.time() - end)
    end = time.time()

    return batch_time.avg, data_time.avg, losses.avg, q_loss.avg, a_loss.avg, G_list, record_matrix



@ex.automain
def main(_run, _config):
    # Deterministic setting for reproducability.
    set_seed(_config["seed"])
    cudnn.deterministic = True
    model = Lobe_fewShotSeg()
    model = model.cuda()

    # Init optimizer.
    optimizer = torch.optim.SGD(model.parameters(), 
                            lr=_config["lr"],
                            momentum=_config["momentum"],
                            weight_decay=_config["weight_decay"])

    scheduler = MultiStepLR(optimizer, milestones=_config["milestones"], gamma=_config["gamma"])

    my_weight = torch.FloatTensor(_config["loss_weights"]).cuda()
    criterion = nn.NLLLoss(ignore_index=_config["ignore_index"], weight=my_weight)

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True

    # Define data set and loader.
    train_dataset = LobeTrainDataset(_config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=_config["batch_size"],
        shuffle=_config["shuffle"],
        num_workers=_config["num_workers"],
        pin_memory=_config["pin_memory"],
        drop_last=_config["drop_last"]
    )
    
    # Start training.
    print('  Start training ...')
    G_list = [[[[] for _ in range(_config["slicer_len"])] for _ in range(len(_config['label_names']))]  for _ in range(2)]
    record_matrix = torch.zeros(5, _config["slicer_len"])

    for epoch in range(_config["epochs"]):
        record = epoch >= _config["record_epochs"]

        # Train.
        batch_time, data_time, losses, q_loss, align_loss, G_list, record_matrix = train(
            train_loader, model, criterion, optimizer, scheduler, G_list, _config["slicer_len"], record, record_matrix)

        # Log
        print('============== Epoch [{}] =============='.format(epoch))
        print('  Batch time: {:6.3f}'.format(batch_time))
        print('  Loading time: {:6.3f}'.format(data_time))
        print('  Total Loss  : {:.5f}'.format(losses))
        print('  Query Loss  : {:.5f}'.format(q_loss))
        print('  Align Loss  : {:.5f}'.format(align_loss))


    # Save trained model.
    save_model(model, G_list, record_matrix, _config)


