
def IoU(pred, ground_truth, smooth=1e-6):
    pred = pred.reshape(pred.shape[0], 5, -1)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], 5, -1)

    intersection = pred * ground_truth
    intersection = intersection.sum(axis=-1)

    total = ground_truth.sum(axis=-1) + pred.sum(axis=-1)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return IoU.mean() * 100
