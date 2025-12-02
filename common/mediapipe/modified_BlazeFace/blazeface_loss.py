# blazeface_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(box_a.size(0), box_b.size(0), 2),
                       box_b[:, 2:].unsqueeze(0).expand(box_a.size(0), box_b.size(0), 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(box_a.size(0), box_b.size(0), 2),
                       box_b[:, :2].unsqueeze(0).expand(box_a.size(0), box_b.size(0), 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def point_form(boxes):
    # Convert (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

class MultiBoxLoss(nn.Module):
    def __init__(self, cfg, overlap_thresh=0.35, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.variance = cfg['variance']
        self.threshold = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio

    def encode(self, matched, priors, keypoints):
        # Dist b/w match center and prior center
        g_cxcy = (matched[:, :2] + matched[:, 2:4])/2 - priors[:, :2]
        # encode variance
        g_cxcy /= (self.variance[0] * priors[:, 2:])
        # Match wh / prior wh
        g_wh = (matched[:, 2:4] - matched[:, 0:2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / self.variance[1]
        
        # Encode Keypoints (normalize by anchor size)
        g_kps = []
        for i in range(6):
            kp_idx = 4 + i*2
            # Offset from prior center / (variance * prior dim)
            kp_xy = (keypoints[:, kp_idx:kp_idx+2] - priors[:, :2]) / (self.variance[0] * priors[:, 2:])
            g_kps.append(kp_xy)
        g_kps = torch.cat(g_kps, 1)
        
        return torch.cat([g_cxcy, g_wh, g_kps], 1)

    def forward(self, predictions, targets, priors):
        """
        predictions: tuple (loc_data, conf_data)
        targets: list of [num_objs, 16] (4 box + 12 kps)
        priors: [num_priors, 4] (cx, cy, w, h)
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :] # Safety check
        num_priors = (priors.size(0))
        num_classes = conf_data.size(2)

        # Match priors with ground truth
        loc_t = torch.Tensor(num, num_priors, 16).to(loc_data.device)
        conf_t = torch.LongTensor(num, num_priors).to(loc_data.device)

        for idx in range(num):
            truths = targets[idx][:, :4]
            kps = targets[idx][:, 4:] # Remaining are keypoints
            labels = torch.ones(len(truths)).to(loc_data.device) # Class 1 for Face
            
            # IoU Matching
            defaults = priors.data
            overlaps = jaccard(truths, point_form(defaults))
            
            best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
            best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
            
            best_truth_idx.squeeze_(0)
            best_truth_overlap.squeeze_(0)
            best_prior_idx.squeeze_(1)
            best_prior_overlap.squeeze_(1)
            
            best_truth_overlap.index_fill_(0, best_prior_idx, 2)
            
            # Assign matches
            matches = truths[best_truth_idx]
            matches_kps = kps[best_truth_idx]
            conf = labels[best_truth_idx] # Shape: [num_priors]
            conf[best_truth_overlap < self.threshold] = 0 # Background
            
            # Encode
            loc_t[idx] = self.encode(matches, defaults, matches_kps)
            conf_t[idx] = conf

        # Hard Negative Mining
        pos = conf_t > 0
        
        # Classification Loss (Cross Entropy)
        # Flatten to (batch*priors, num_classes)
        batch_conf = conf_data.view(-1, num_classes)
        # We need BCEWithLogits for binary, but let's assume standard CrossEntropy logic for expandability
        # If output is 1-dim (score), use BCE. If 2-dim (bg, face), use CE.
        # Here we did output size 1. 
        loss_c = F.binary_cross_entropy_with_logits(conf_data.squeeze(), conf_t.float(), reduction='none')
        
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0 # filter out positives
        
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg

        # Final Loss Calculation
        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 16)
        loc_t = loc_t[pos_idx].view(-1, 16)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Concat pos and neg for class loss
        pos_mask = pos.unsqueeze(2).expand_as(conf_data)
        neg_mask = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_mask+neg_mask).gt(0)].view(-1, 1)
        targets_weighted = conf_t[(pos+neg).gt(0)].float().view(-1, 1)
        loss_c = F.binary_cross_entropy_with_logits(conf_p, targets_weighted, reduction='sum')

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c