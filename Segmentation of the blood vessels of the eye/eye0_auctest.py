import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import auc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def nor_img(img):
    return (img - img.min()) / (img.max() - img.min())

def patch(image, patch_size, step):
    xs = list(range(0, image.shape[0] - patch_size[0], step)) + [image.shape[0] - patch_size[0]]
    ys = list(range(0, image.shape[1] - patch_size[1], step)) + [image.shape[1] - patch_size[1]]
    patch_list = []
    loc_list = []
    for x in xs:
        for y in ys:
            patch_list += [np.expand_dims(image[x:x + patch_size[0], y:y + patch_size[1], :], axis=0)]
            loc_list += [[x, y]]
    patches = np.concatenate(tuple(patch_list), axis=0)
    return patches, loc_list

def test(model, patches, loc_list, image_shape, batch_size, img_id):
    batch_num = patches.shape[0] // batch_size
    patches_out = np.zeros(patches.shape[:-1] + (2,), dtype=np.float32)
    image_out = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)
    for n in range(batch_num + 1):
        print('{}: {}/{}'.format(img_id, n, batch_num + 1), end='\r')
        out = model(patches[n * batch_size:(n + 1) * batch_size, :, :, :]).numpy()
        patches_out[n * batch_size:(n + 1) * batch_size, :, :, :] = out
    for i in range(patches_out.shape[0]):
        x, y = loc_list[i]
        image_out[x + 64:x + patches_out.shape[1] - 64, y + 64:y + patches_out.shape[2] - 64] += patches_out[i, 64:-64, 64:-64, 1]
        weight[x + 64:x + patches_out.shape[1] - 64, y + 64:y + patches_out.shape[2] - 64] += 1
    image_out = image_out / (weight + 0.001)
    return image_out

def evaluate(seg, gt):
    tp = (seg & gt).sum()
    tn = ((1 - seg) & (1 - gt)).sum()
    fp = (seg & (1 - gt)).sum()
    fn = ((1 - seg) & gt).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return tp, tn, fp, fn, acc, sen, spe

def main():
    seg_dir = "image/{}/{}.1.jpg"
    img_dir = "image/{}/{}.jpg"

    for m in range(22, 23):
        model = load_model('/home/gem/Harry/DRIVE_model_Augment_model/model_DRIVE_Augment_22')
        acc_list = []
        sen_list = []
        spe_list = []
        auc_scores = []
        all_fpr = []
        all_tpr = []

        if str(m) not in os.listdir('/home/gem/Harry/0411/svd_aug0.8001'):
            os.mkdir('/home/gem/Harry/0411/svd_aug0.8001/{}'.format(m))
        with open('/home/gem/Harry/0411/svd_aug0.8001/{}/res_rec_{}.csv'.format(m, m), 'w') as f:
            f.write('ID\tTP\tTN\tFP\tFN\n')

        for i in range(151, 201):  # Testing dataset
            if i == 186:
                continue  # 跳过处理第186张图片
            img = cv2.imread(img_dir.format(i, i))
            img = nor_img(img)
            seg = cv2.imread(seg_dir.format(i, i))
            seg = seg.min(axis=-1)
            seg = np.array(1 * (seg > 230), dtype=np.uint8)

            patches, loc = patch(img, (512, 512), 256)
            image_out = test(model, patches, loc, img.shape[:-1], 1, i)
            image_out_flat = image_out.flatten()
            seg_flat = seg.flatten()

            fpr, tpr, thresholds = roc_curve(seg_flat, image_out_flat)
            all_fpr.append(fpr)
            all_tpr.append(tpr)

            auc_score = roc_auc_score(seg_flat, image_out_flat)
            auc_scores.append(auc_score)  # 将AUC分数添加到列表中
            #print('AUC for image {}: {:.4f}'.format(i, auc_score))

            cv2.imwrite('/home/gem/Harry/0411/svd_aug0.8001/{}/res_{}.png'.format(m, i), np.hstack((image_out, seg)) * 255)

            image_out = np.array(1 * (image_out > 0.5), dtype=np.uint8)
            tp, tn, fp, fn, acc, sen, spe = evaluate(image_out, seg)
            with open('/home/gem/Harry/0411/svd_aug0.8001/{}/res_rec_{}.csv'.format(m, m), 'a') as f:
                f.write('{}\t{}\t{}\t{}\t{}\tAUC={:.4f}\n\n'.format(i, tp, tn, fp, fn, auc_score))

            acc_list.append(acc)
            sen_list.append(sen)
            spe_list.append(spe)
        
        # 所有图像处理之后，绘制平均ROC曲线
        # 创建一个所有FPR值的合并列表
        all_fpr_unique = np.unique(np.concatenate(all_fpr))
        
        # 然后插值所有TPR值在这个点集上
        mean_tpr = np.zeros_like(all_fpr_unique)
        for i in range(len(all_tpr)):
            mean_tpr += np.interp(all_fpr_unique, all_fpr[i], all_tpr[i])
        
        # 计算平均TPR
        mean_tpr /= len(all_tpr)
        
        # 计算AUC
        mean_auc = auc(all_fpr_unique, mean_tpr)
        
        # 绘制平均ROC曲线
        plt.figure(figsize=(10, 6))
        plt.plot(all_fpr_unique, mean_tpr, color='orange', label='ROC curve (area = {:.2f})'.format(mean_auc), lw=2)
        acc_list += [acc]
        sen_list += [sen]
        spe_list += [spe]
        print('Model: {}'.format(m),
              'acc:', np.array(acc_list).mean(),
              'sen:', np.array(sen_list).mean(),
              'spe:', np.array(spe_list).mean())

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'b--', lw=2)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('/home/gem/Harry/0411/svd_aug0.8001/{}/roc_curve.png'.format(m))
        plt.show()

        # AUC平均值的计算
        if auc_scores:
            auc_average = sum(auc_scores) / len(auc_scores)
            print('Average AUC for model {}: {:.4f}'.format(m, auc_average))
        else:
            print('No AUC scores available to calculate average.')
        

if __name__ == '__main__':
    main()