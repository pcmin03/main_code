import numpy as np

# Loss Meter
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self,num_class):
        self.reset()
        self.reset_dict()
        self.num_class = num_class

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset_dict(self):
        self.IOU_scalar = dict()
        self.precision_scalar = dict()
        self.recall_scalr = dict()
        self.F1score_scalar = dict()

    def update_dict(self,result_dicts):

        for i in range(self.num_class):
            self.IOU_scalar.update({'IOU_'+str(i):result_dicts['IOU'][i]})
            self.precision_scalar.update({'precision_'+str(i):result_dicts['precision'][i]})
            self.recall_scalr.update({'recall_'+str(i):result_dicts['recall'][i]})
            self.F1score_scalar.update({'F1_'+str(i):result_dicts['F1'][i]})
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        # print(self.confusion_matrix.shape)
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Class_Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Class_Acc)
        wo_back_ACC = np.nanmean(Class_Acc[1:self.num_class])
        return Acc,Class_Acc,wo_back_ACC

    def Mean_Intersection_over_Union(self):
        Class_MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(Class_MIoU)
        wo_back_MIoU = np.nanmean(Class_MIoU[1:self.num_class])
        return MIoU,Class_MIoU,wo_back_MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    
    def Class_F1_score(self):
        Class_precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        Class_recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        Class_F1score = 2*((Class_precision*Class_recall)/(Class_precision+Class_recall))
        return Class_precision, Class_recall,Class_F1score
    
    def Class_Fbeta_score(self,beta):
        beta2 = np.sqrt(beta)
        Class_precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        Class_recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        Class_F1score = (1+beta2)*((Class_precision*Class_recall)/(beta2*Class_precision+Class_recall))
        return Class_precision, Class_recall,Class_F1score

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        # print(confusion_matrix)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # print(gt_image.shape,pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def update(self): 
        _,self.Class_IOU,_ = self.Mean_Intersection_over_Union()
        # Acc_class,Class_ACC,wo_back_ACC = self.Pixel_Accuracy_Class()
        self.Class_precision, self.Class_recall,self.Class_F1score = self.Class_F1_score()
        # _, _,Class_Fbetascore = self.Class_Fbeta_score(beta=betavalue)

        total_dict = {'IOU':self.Class_IOU,
                    'precision':self.Class_precision,
                    'recall':self.Class_recall,
                    'F1':self.Class_F1score}

        return total_dict

        # total_dict = {'IOU':Class_IOU}
        # # for i in range(classnum):
        # IOU_scalar.update({'val_IOU_'+str(i):Class_IOU[i]})
        # precision_scalar.update({'val_precision_'+str(i):Class_precision[i]})
        # recall_scalr.update({'val_recall_'+str(i):Class_recall[i]})
        # F1score_scalar.update({'val_F1_'+str(i):Class_F1score[i]})
        # Fbetascore_scalar.update({'val_Fbeta'+str(i):Class_Fbetascore[i]})