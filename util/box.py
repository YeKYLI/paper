
def compute_iou(x1min, y1min, x1max, y1max, x2min, y2min, x2max, y2max):
    if x1min > x2max or x1max < x2min:
        return 0
    else:
        xmin = max(x1min, x2min)
        xmax = min(x1max, x2max)
    if y1min > y2max or y1max < y2min:
        return 0
    else:
        ymin = max(y1min, y2min)
        ymax = min(y1max, y2max)
    area_1 = (x1max - x1min) * (y1max - y1min)
    area_2 = (x2max - x2min) * (y2max - y2min)
    area_iou = (xmax - xmin) * (ymax - ymin)
    iou = float(area_iou) / float(area_1 + area_2 - area_iou)
    return iou

