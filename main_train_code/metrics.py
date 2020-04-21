import numpy as np

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

