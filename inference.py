import pickle
import pandas as pd
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from models.lobe_fewshot import Lobe_fewShotSeg
from dataloading.datasets import LobeTestDataset, Lobe_inference
from tqdm import tqdm
from utils import *
from config import ex 


def infer(model, query_loader, label_names, G_P, index_m, _config):
    scores = Scores(len(label_names) - 1)
    for i, sample in enumerate(query_loader):
        # Extract query images.
        query_images = sample['rs_image'].float().cuda()  # [N x C x H x W]
        query_label = sample['label'].long()  # C x H x W   testing only
        rs_len = sample['rs_len'][0].numpy()  
        # filename = sample['filename'] 
        # direction = sample['direction']
        # spacing = sample['spacing'], 
        # origin = sample['origin'],

        num_slices = query_images.shape[1]

        preds = []
        print(f"-----Sample {i}-----")

        for slice_idx in tqdm(range(num_slices)):
            qry_img = [query_images[:, slice_idx, :, :]]  # [1 x C x H x W]
            pred = torch.zeros(6, 512, 512).cuda()  # Initialize prediction tensor

            pscale_factor = _config["slicer_len"] / num_slices

            for label_name in label_names:
                if label_name == 'BG':
                    continue

                key = int(slice_idx * pscale_factor)
                label_key = int(label_name)

                # Retrieve BG and FG data
                BG_proto, FG_proto, alp = retrieve_prototype(G_P, index_m, key, label_key, _config)

                if alp:
                    query_pred, *_ = model(None, None, None, qry_img, BG_proto, FG_proto, alp, isval=True)
                    query_pred = [F.softmax(pred.squeeze(0), dim=0) for pred in query_pred]                           
                    pred[0] = query_pred[0][0].clone()
                    pred[label_key] = query_pred[1][1].clone()
            preds.append(pred)

        # Convert prediction tensor to final output
        preds = torch.stack(preds)
        preds = preds.argmax(dim=1)
        preds = mode_filter_processing(preds)
        _, preds = data_resize(query_images[0].permute(1, 0, 3, 2).cpu().numpy(), preds.cpu().numpy(), rs_len)
        # saving(preds.transpose(2, 1, 0), filename[0], _config, direction, spacing, origin)
        
        # testing only
        scores.record(torch.from_numpy(preds).cuda(), query_label.squeeze(0).cuda())
    scores.average_patient_dice()
    scores.average_patient_iou()


def retrieve_prototype(G_P, index_m, key, label_key, _config):
    """
    Retrieve background and foreground prototype data based on key and indices.
    """
    nkeys, alp = get_proto_index(index_m, int(label_key) - 1, int(key), _config["SCCGRD_len"], _config["SCCGRD_alp"])
    BG_proto = []
    FG_proto = []
    for nkey in nkeys:
        BG_proto.append([torch.from_numpy(G_P[0][0][nkey]).cuda(), torch.from_numpy(G_P[0][label_key][nkey]).cuda()])
        FG_proto.append([torch.from_numpy(G_P[1][0][nkey]).cuda(), torch.from_numpy(G_P[1][label_key][nkey]).cuda()])
    return BG_proto, FG_proto, alp
    

@ex.automain
def main(_run, _config):
    # Set seed and deterministic behavior
    set_seed(_config["seed"])
    cudnn.deterministic = True

    # Initialize model and load weights
    model = Lobe_fewShotSeg().cuda()
    model.load_state_dict(torch.load(_config["model_path"], map_location="cpu"))
    model.eval()

    # Load G_P and index_m
    with open(_config["G_path"], 'rb') as g_path:
        G_P = pickle.load(g_path)
    index_m = pd.read_csv(_config["index_m_path"], header=None).to_numpy()

    # Data loader setup
    test_dataset = LobeTestDataset(_config)
    query_loader = DataLoader(
        test_dataset,
        batch_size=_config["batch_size"],
        shuffle=_config["shuffle"],
        num_workers=_config["num_workers"],
        pin_memory=_config["pin_memory"],
        drop_last=_config["drop_last"]
    )

    # Class labels
    label_names = _config['label_names']
    print('  *--------------------Beginning---------------------*')

    # Inference
    with torch.no_grad():
        infer(model, query_loader, label_names, G_P, index_m, _config)
