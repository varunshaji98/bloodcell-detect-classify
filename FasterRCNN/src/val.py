import utils
from utils import vocpascal2bb, MethodAveragePrecision
from bbox_metrics import get_pascalvoc_metrics, plot_precision_recall_curves, plot_precision_recall_curves_all

det_boxes = vocpascal2bb('../test_predictions', 'DETECTED')
gt_boxes = vocpascal2bb('../data/test', 'GT')

result = get_pascalvoc_metrics(gt_boxes,
                    det_boxes,
                    iou_threshold=0.7,
                    method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                    generate_table=True)

print(result['mAP'])
print(result['per_class']['rbc'])
print(result['per_class']['wbc'])

out_file = open('outputs/results.txt', 'w')
out_file.write(str(result))

plot_precision_recall_curves_all(result['per_class'], showInterpolatedPrecision=True, showAP=True)

