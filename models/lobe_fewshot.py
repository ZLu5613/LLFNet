import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.torchvision_backbones import TVDeeplabRes101Encoder

class Lobe_fewShotSeg(nn.Module):
    def __init__(self, n_ways=1, n_shots=1, n_queries=1, sup_bsize=1):
        super(Lobe_fewShotSeg, self).__init__()
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.sup_bsize = sup_bsize
        self.thresh = 0.95
        self.encoder = TVDeeplabRes101Encoder()

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, 
                BG_proto=[None, None], FG_proto=[None, None], alp=[], isval=False):
        
        if self.training:
            return self.training_step(supp_imgs, fore_mask, back_mask, qry_imgs, BG_proto, FG_proto)
        else:
            return self.validation_step(qry_imgs, BG_proto, FG_proto, alp)
        

    def training_step(self, supp_imgs, fore_mask, back_mask, qry_imgs, BG_proto, FG_proto):
        """
        Handles the forward pass during training.
        """
        n_ways, n_shots, sup_bsize, n_queries = self.n_ways, self.n_shots, self.sup_bsize, self.n_queries
        img_size = qry_imgs[0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        # Concatenate images
        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0
        )

        # Convert masks 
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        # Feature extraction
        img_fts = self.encoder(imgs_concat.float(), low_level=False)
        fts_size = img_fts.shape[-2:]
        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(n_ways, n_shots, sup_bsize, -1, *fts_size)
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(n_queries, qry_bsize, -1, *fts_size)

        align_loss = 0
        outputs = [None, None]

        for ways in range(2):
            output = []
            scores = []

            res_fg_msk = torch.stack(
                [F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') 
                 for fore_mask_w in fore_mask[:, ways].unsqueeze(0)], dim=0)
            res_bg_msk = torch.stack(
                [F.interpolate(back_mask_w, size=fts_size, mode='bilinear') 
                 for back_mask_w in back_mask[:, ways].unsqueeze(0)], dim=0)

            # Compute prototypes
            BG_proto[ways] = self.get_bg_Proto(qry_fts, supp_fts, res_bg_msk, thresh = self.thresh)
            _raw_score = self.get_pred(qry_fts, BG_proto[ways])
            scores.append(_raw_score)

            FG_proto[ways] = self.get_fg_Proto(qry_fts, supp_fts, res_fg_msk, thresh = self.thresh)
            _raw_score = self.get_pred(qry_fts, FG_proto[ways])
            scores.append(_raw_score)

            # Compute loss
            pred = torch.cat(scores, dim=1)
            align_loss_epi = self.alignLoss(qry_fts, pred, supp_fts, fore_mask[:, ways], back_mask[:, ways])
            align_loss += align_loss_epi

            output.append(F.interpolate(pred, size=img_size, mode='bilinear'))
            output = torch.stack(output, dim=1)
            outputs[ways] = output.view(-1, *output.shape[2:])
        return outputs, align_loss, BG_proto, FG_proto

    def validation_step(self, qry_imgs, BG_proto, FG_proto, alp):
        """
        Handles the forward pass during validation.
        """
        n_ways, n_shots, sup_bsize, n_queries = self.n_ways, self.n_shots, self.sup_bsize, self.n_queries
        img_size = qry_imgs[0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        # Concatenate query images twice (for consistency)
        imgs_concat = torch.cat([torch.cat(qry_imgs, dim=0)] * 2, dim=0)

        # Feature extraction
        img_fts = self.encoder(imgs_concat.float(), low_level=False)
        fts_size = img_fts.shape[-2:]
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(n_queries, qry_bsize, -1, *fts_size)

        outputs = [None, None]

        for ways in range(2):
            bg_list, fg_list = [], []

            for i, alpha in enumerate(alp):
                raw_bg = self.get_pred(qry_fts, BG_proto[i][ways])
                raw_fg = self.get_pred(qry_fts, FG_proto[i][ways])

                bg_list.append(raw_bg * alpha)
                fg_list.append(raw_fg * alpha)

            # Aggregate weighted predictions
            bg_weighted = torch.stack(bg_list, dim=0).sum(dim=0)
            fg_weighted = torch.stack(fg_list, dim=0).sum(dim=0)

            pred = torch.cat([bg_weighted, fg_weighted], dim=1)
            output = F.interpolate(pred, size=img_size, mode='bilinear')

            outputs[ways] = output.view(-1, *output.shape[1:])
        return outputs, 0, BG_proto, FG_proto
    

    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(2)]

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)
        
        ### end of added part
        img_fts = supp_fts # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

        qry_pred_fg_msk = F.interpolate(binary_masks[1].float(), size = img_fts.shape[-2:], mode = 'bilinear') # [1 (way), n (shot), h, w]
        qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear') # 1, n, h ,w
    
        scores = []

        bg_proto = self.get_bg_Proto(img_fts.squeeze(0), qry_fts.squeeze(0), qry_pred_bg_msk, self.thresh)
        _raw_score_bg = self.get_pred(img_fts.squeeze(0), bg_proto)
        scores.append(_raw_score_bg)

        fg_proto = self.get_fg_Proto(img_fts.squeeze(0), qry_fts.squeeze(0), qry_pred_fg_msk, self.thresh)
        _raw_score_fg = self.get_pred(img_fts.squeeze(0), fg_proto)
        scores.append(_raw_score_fg)

        supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
        supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

        # Construct the support Ground-Truth segmentation
        supp_label = torch.full_like(fore_mask, 255, device=img_fts.device).long()
        supp_label[fore_mask == 1] = 1
        supp_label[back_mask == 1] = 0

        # Compute Loss
        return F.cross_entropy(supp_pred, supp_label.squeeze(0), ignore_index=255)

    
    def get_bg_Proto(self, qry, sup_x, sup_y, thresh, kernel_size = [4, 4]):
        avg_pool_op = torch.nn.AvgPool2d(kernel_size)
        
        # Process support features
        sup_x, qry, sup_y = sup_x.squeeze(0).squeeze(1), qry.squeeze(1), sup_y.squeeze(0)
        nch, sup_nshot = qry.shape[1], sup_x.shape[0]

        # Average pool and reshape support features
        n_sup_x = avg_pool_op(sup_x)
        n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

        # Average pool and reshape support labels
        sup_y_g = avg_pool_op(sup_y)
        sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

        # Generate prototypes
        protos = self.safe_norm(n_sup_x[sup_y_g > thresh, :])  # [n_pro, nc]
        return protos
        
    def get_fg_Proto(self, qry, sup_x, sup_y, thresh, kernel_size = [4, 4]):
        avg_pool_op = torch.nn.AvgPool2d(kernel_size)

        # Process support features
        sup_x, qry, sup_y = sup_x.squeeze(0).squeeze(1), qry.squeeze(1), sup_y.squeeze(0)
        nch, sup_nshot = qry.shape[1], sup_x.shape[0]

        # Average pool and reshape support features
        n_sup_x = avg_pool_op(sup_x)
        n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

        # Average pool and reshape support labels
        sup_y_g = avg_pool_op(sup_y)
        sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

        # Generate prototypes
        protos = n_sup_x[sup_y_g > thresh, :]  # [n_pro, nc]

        # Compute global prototype
        glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)

        protos = self.safe_norm(torch.cat( [protos, glb_proto], dim = 0 ))
        return protos
    def safe_norm(self, x, p = 2, dim = 1, eps = 1e-4):
        x_norm = torch.norm(x, p = p, dim = dim) # .detach()
        x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
        x = x.div(x_norm.unsqueeze(1).expand_as(x))
        return x    
    
    def get_pred(self, features, proto):
        features = self.safe_norm(features.squeeze(1))
        dists = F.conv2d(features, proto[..., None, None]) * 20
        pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
        return pred_grid

