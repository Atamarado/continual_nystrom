import numpy as np

def frame_level_map_n_cap(results):
    all_probs = results['probs']
    all_labels = results['labels']   

    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    for i in range(1, n_classes):
        this_cls_prob = all_probs[i, :]
        this_cls_gt = all_labels[i, :]
        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0., 0.
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)
        this_cls_ap = psum/np.sum(this_cls_gt)
        this_cls_acp = cpsum / np.sum(this_cls_gt)

        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    map = sum(all_cls_ap) / len(all_cls_ap)
    cap = sum(all_cls_acp) / len(all_cls_acp)
    return map, all_cls_ap, cap, all_cls_acp