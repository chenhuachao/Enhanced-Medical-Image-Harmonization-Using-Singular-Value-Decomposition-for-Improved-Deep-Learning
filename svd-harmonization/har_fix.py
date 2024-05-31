# 图片分解成RGB三个通道，然后分别显示。
import numpy as np
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
from scipy import misc
import cv2


def imgCompress(channel, percent):
    U, sigma, V_T = np.linalg.svd(channel)
    m = U.shape[0]
    n = V_T.shape[0]
    reChannel = np.zeros((m, n))

    for k in range(len(sigma)):
        reChannel = reChannel + sigma[k] * np.dot(U[:, k].reshape(m, 1), V_T[k, :].reshape(1, n))
        if float(k) / len(sigma) > percent:
            reChannel[reChannel < 0] = 0
            reChannel[reChannel > 255] = 255
            break
    return np.rint(reChannel).astype("uint8")
    # return np.rint(reChannel).astype(np.float32)  
    #return reChannel

def imgHandle(img_path):
    im = Image.open(img_path).convert("RGB")
    # 拆分通道
    r, g, b = im.split()
    #print(np.array(r).tolist())
    oriImage = [r, g, b]
    imgArray = []
    img_r_array = []
    img_g_array = []
    img_b_array = []
    
    for i in range(3):
        imgArray.append(np.array(oriImage[i], dtype=np.float32))  # 指定dtype为float32

    for i in range(3):
        for p in [0.1, 0.3, 0.5]:
            re = imgCompress(imgArray[i], p)
            # Image.fromarray(re).save("channel_breast_{}_{}.png".format(i, int(p)))
            # Image.fromarray(re).save("{}".format(p)+"in_H.jpg")
            if i == 0:
                img_r_array.append(re)
            elif i == 1:
                img_g_array.append(re)
            elif i == 2:
                img_b_array.append(re)

    # 图片相减去
    img_r1 = cv2.subtract(img_r_array[2], img_r_array[1])
    img_r2 = cv2.subtract(img_r_array[1], img_r_array[0])
    # print(np.array(r).dtype)
    # print(img_r_array[2])
    img_r3 = cv2.subtract(np.array(r), img_r_array[2])

    img_g1 = cv2.subtract(img_g_array[2], img_g_array[1])
    img_g2 = cv2.subtract(img_g_array[1], img_g_array[0])
    img_g3 = cv2.subtract(np.array(g), img_g_array[2])

    img_b1 = cv2.subtract(img_b_array[2], img_b_array[1])
    img_b2 = cv2.subtract(img_b_array[1], img_b_array[0])
    img_b3 = cv2.subtract(np.array(b), img_b_array[2])



    r1_std = ImageStat.Stat(Image.fromarray(img_r1)).stddev
    r1_mean = ImageStat.Stat(Image.fromarray(img_r1)).mean

    r2_std = ImageStat.Stat(Image.fromarray(img_r2)).stddev
    r2_mean = ImageStat.Stat(Image.fromarray(img_r2)).mean

    r3_std = ImageStat.Stat(Image.fromarray(img_r3)).stddev
    r3_mean = ImageStat.Stat(Image.fromarray(img_r3)).mean

    g1_std = ImageStat.Stat(Image.fromarray(img_g1)).stddev
    g1_mean = ImageStat.Stat(Image.fromarray(img_g1)).mean

    g2_std = ImageStat.Stat(Image.fromarray(img_g2)).stddev
    g2_mean = ImageStat.Stat(Image.fromarray(img_g2)).mean

    g3_std = ImageStat.Stat(Image.fromarray(img_g3)).stddev
    g3_mean = ImageStat.Stat(Image.fromarray(img_g3)).mean

    b1_std = ImageStat.Stat(Image.fromarray(img_b1)).stddev
    b1_mean = ImageStat.Stat(Image.fromarray(img_b1)).mean

    b2_std = ImageStat.Stat(Image.fromarray(img_b2)).stddev
    b2_mean = ImageStat.Stat(Image.fromarray(img_b2)).mean

    b3_std = ImageStat.Stat(Image.fromarray(img_b3)).stddev
    b3_mean = ImageStat.Stat(Image.fromarray(img_b3)).mean

    r_std_007 = ImageStat.Stat(Image.fromarray(img_r_array[0])).stddev
    r_mean_007 = ImageStat.Stat(Image.fromarray(img_r_array[0])).mean

    g_std_007 = ImageStat.Stat(Image.fromarray(img_g_array[0])).stddev
    g_mean_007 = ImageStat.Stat(Image.fromarray(img_g_array[0])).mean

    b_std_007 = ImageStat.Stat(Image.fromarray(img_b_array[0])).stddev
    b_mean_007 = ImageStat.Stat(Image.fromarray(img_b_array[0])).mean
    img_data = [img_r1, img_r2, img_g1, img_g2, img_b1, img_b2, r1_std, r1_mean,
                r2_std, r2_mean, g1_std, g1_mean, g2_std, g2_mean, b1_std, b1_mean,
                b2_std, b2_mean, r_std_007, r_mean_007, g_std_007, g_mean_007,
                b_std_007, b_mean_007, r, g, b, img_r_array[0], img_g_array[0], img_b_array[0]
                , img_r3, img_g3, img_b3, r3_std, r3_mean, g3_std, g3_mean, b3_std, b3_mean]
    return img_data

def imgMerge(img_in, img_re):
    # 输出图片r
    Out_img_r007 = np.array(img_in[27]) - np.array(img_in[19])
    Out_img_r007 *= np.array(img_re[18])
    Out_img_r007 /= np.array(img_in[18])
    Out_img_r007 += np.array(img_re[19])
    # Out_img_r007 = Out_img_r007.astype(np.float32)
    Out_img_r007 = Out_img_r007.astype(np.float32)
    # img_007 = Image.fromarray(np.uint8(Out_img_in007))

    Out_img_r5 = np.array(img_in[0]) - np.array(img_in[7])
    Out_img_r5 *= np.array(img_re[6])
    Out_img_r5 /= np.array(img_in[6])
    Out_img_r5 += np.array(img_re[7])
    # Out_img_r5 = Out_img_r5.astype(np.float32)
    Out_img_r5 = Out_img_r5.astype(np.float32)

    # img_5 = Image.fromarray(np.uint8(Out_img_5))

    Out_img_r6 = np.array(img_in[1]) - np.array(img_in[9])
    Out_img_r6 *= np.array(img_re[8])
    Out_img_r6 /= np.array(img_in[8])
    Out_img_r6 += np.array(img_re[9])
    # Out_img_r6 = Out_img_r6.astype(np.float32)
    Out_img_r6 = Out_img_r6.astype(np.float32)
    # img_6 = Image.fromarray(np.uint8(Out_img_6))

    Out_img_r7 = np.array(img_in[30]) - np.array(img_in[34])
    Out_img_r7 *= np.array(img_re[33])
    Out_img_r7 /= np.array(img_in[33])
    Out_img_r7 += np.array(img_re[34])
    # Out_img_r7 = Out_img_r7.astype(np.float32)
    Out_img_r7 = Out_img_r7.astype(np.float32)

    # cv2.imwrite("out_img_r_007.jpg", Out_img_r007)
    # cv2.imwrite("out_img_r_5.png", Out_img_r5)
    # cv2.imwrite("out_img_r_6.png", Out_img_r6)
    # cv2.imwrite("out_img_r_7.png", Out_img_r7)

    Out_img_r_1 = cv2.add(cv2.add(Out_img_r5, Out_img_r6), Out_img_r7)
    Out_img_r = cv2.add(Out_img_r_1, Out_img_r007)

    # cv2.imwrite("11111.jpg", Out_img_r)

    # 输出图片g
    Out_img_g007 = np.array(img_in[28]) - np.array(img_in[21])
    Out_img_g007 *= np.array(img_re[20])
    Out_img_g007 /= np.array(img_in[20])
    Out_img_g007 += np.array(img_re[21])
    # Out_img_g007 = Out_img_g007.astype(np.float32)
    Out_img_g007 = Out_img_g007.astype(np.float32)
    # img_007 = Image.fromarray(np.uint8(Out_img_in007))
    # cv2.imwrite("out_img_g_007.jpg", Out_img_g007)

    Out_img_g5 = np.array(img_in[2]) - np.array(img_in[11])
    Out_img_g5 *= np.array(img_re[10])
    Out_img_g5 /= np.array(img_in[10])
    Out_img_g5 += np.array(img_re[11])
    # Out_img_g5 = Out_img_g5.astype(np.float32)
    Out_img_g5 = Out_img_g5.astype(np.float32)
    # img_5 = Image.fromarray(np.uint8(Out_img_5))

    Out_img_g6 = np.array(img_in[3]) - np.array(img_in[13])
    Out_img_g6 *= np.array(img_re[12])
    Out_img_g6 /= np.array(img_in[12])
    Out_img_g6 += np.array(img_re[13])
    # Out_img_g6 = Out_img_g6.astype(np.float32)
    Out_img_g6 = Out_img_g6.astype(np.float32)
    # img_6 = Image.fromarray(np.uint8(Out_img_6))

    Out_img_g7 = np.array(img_in[31]) - np.array(img_in[36])
    Out_img_g7 *= np.array(img_re[35])
    Out_img_g7 /= np.array(img_in[35])
    Out_img_g7 += np.array(img_re[36])
    # Out_img_g7 = Out_img_g7.astype(np.float32)
    Out_img_g7 = Out_img_g7.astype(np.float32)

    Out_img_g_1 = cv2.add(cv2.add(Out_img_g5, Out_img_g6), Out_img_g7)
    Out_img_g = cv2.add(Out_img_g_1, Out_img_g007)

                        
    # cv2.imwrite("22222.jpg", Out_img_g)

    # 输出图片b
    Out_img_b007 = np.array(img_in[29]) - np.array(img_in[23])
    Out_img_b007 *= np.array(img_re[22])
    Out_img_b007 /= np.array(img_in[22])
    Out_img_b007 += np.array(img_re[23])
    # Out_img_b007 = Out_img_b007.astype(np.float32)
    Out_img_b007 = Out_img_b007.astype(np.float32)
    # cv2.imwrite("out_img_b_007.jpg", Out_img_b007)
    # img_007 = Image.fromarray(np.uint8(Out_img_in007))

    Out_img_b5 = np.array(img_in[4]) - np.array(img_in[15])
    Out_img_b5 *= np.array(img_re[14])
    Out_img_b5 /= np.array(img_in[14])
    Out_img_b5 += np.array(img_re[15])
    # Out_img_b5 = Out_img_b5.astype(np.float32)
    Out_img_b5 = Out_img_b5.astype(np.float32)
    # img_5 = Image.fromarray(np.uint8(Out_img_5))

    Out_img_b6 = np.array(img_in[5]) - np.array(img_in[17])
    Out_img_b6 *= np.array(img_re[16])
    Out_img_b6 /= np.array(img_in[16])
    Out_img_b6 += np.array(img_re[17])
    # Out_img_b6 = Out_img_b6.astype(np.float32)
    Out_img_b6 = Out_img_b6.astype(np.float32)
    # img_6 = Image.fromarray(np.uint8(Out_img_6))

    Out_img_b7 = np.array(img_in[32]) - np.array(img_in[38])
    Out_img_b7 *= np.array(img_re[37])
    Out_img_b7 /= np.array(img_in[37])
    Out_img_b7 += np.array(img_re[38])
    # Out_img_b7 = Out_img_b7.astype(np.float32)
    Out_img_b7 = Out_img_b7.astype(np.float32)

    Out_img_b_1 = cv2.add(cv2.add(Out_img_b5, Out_img_b6), Out_img_b7)
    Out_img_b = cv2.add(Out_img_b_1, Out_img_b007)


    # cv2.imwrite("33333.jpg", Out_img_b)
    Out_img_r = np.clip(Out_img_r, 0, 255).astype(np.uint8)
    Out_img_g = np.clip(Out_img_g, 0, 255).astype(np.uint8)
    Out_img_b = np.clip(Out_img_b, 0, 255).astype(np.uint8)

    # Out_img_r007 = np.clip(Out_img_r007, 0, 255).astype(np.uint8)
    # Out_img_g007 = np.clip(Out_img_g007, 0, 255).astype(np.uint8)
    # Out_img_b007 = np.clip(Out_img_b007, 0, 255).astype(np.uint8)
    
    # new_img_007 = Image.merge('RGB', [Image.fromarray(Out_img_r007).convert('L'), Image.fromarray(Out_img_g007).convert('L'),
    #                                   Image.fromarray(Out_img_b007).convert('L')])
    # cv2.imwrite("new_img_007.png", np.array(new_img_007))

    # 合并回RGB三通道
    # img1 = Image.open('./11111.jpg')
    # img2 = Image.open('./22222.jpg')
    # img3 = Image.open('./33333.jpg')
    new = Image.merge('RGB', [Image.fromarray(Out_img_r).convert('L'), Image.fromarray(Out_img_g).convert('L'),
                              Image.fromarray(Out_img_b).convert('L')])
    

    return new