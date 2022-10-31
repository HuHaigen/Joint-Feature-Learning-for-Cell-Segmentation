import os
f1 = open("/home/hhg/dc_pycharm_space/flowerdata/img_path/Image2_train.txt", "w+")

f2 = open("/home/hhg/dc_pycharm_space/flowerdata/img_path/Image2_val.txt", "w+")

# f3 = open("/home/hhg/dc_pycharm_space/flowerdata/img_path/Image10_2_-f-b.txt", "w+")
train_path = "/home/hhg/dc_pycharm_space/flowerdata/ImageNetlike/train"
val_path = "/home/hhg/dc_pycharm_space/flowerdata/ImageNetlike/val"
# bf_path = "/home/hhg/dc_pycharm_space/flowerdata/ImageNet10_2/-b-f"


imgs = os.listdir(train_path)
# print(imgs)
for i in range(len(imgs)):
    f1.writelines(os.path.join(train_path, imgs[i] + "\n"))
    print(imgs[i])

imgs = os.listdir(val_path)
# print(imgs)
for i in range(len(imgs)):
    f2.writelines(os.path.join(val_path, imgs[i] + "\n"))

# imgs = os.listdir(bf_path)
# # print(imgs)
# for i in range(len(imgs)):
#     f3.writelines(os.path.join(bf_path, imgs[i] + "\n"))

f1.close()
f2.close()
# f3.close()

# f1 = open("/home/hhg/dc_pycharm_space/flowerdata/img_path/train_4.txt", "w+")
# f2 = open("/home/hhg/dc_pycharm_space/flowerdata/img_path/val_4.txt", "w+")
#
# val_path = "/home/hhg/dc_pycharm_space/flowerdata/ImageNet10_2/val"
# train_path = "/home/hhg/dc_pycharm_space/flowerdata/ImageNet10_2/train"
# imgs = os.listdir(train_path)
# for i in range(len(imgs)):
#     f1.writelines(os.path.join(train_path, imgs[i] + "\n"))
# imgs = os.listdir(val_path)
# for i in range(len(imgs)):
#     if i % 25 == 0:
#         f1.writelines(os.path.join(train_path, imgs[i] + "\n"))
#     else:
#         f2.writelines(os.path.join(val_path, imgs[i] + "\n"))
# f1.close()
# f2.close()