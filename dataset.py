import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self,data_root_dir,LR_root_dir,HR_root_dir):
        super(MyImageFolder, self).__init__()
        self.data_LR = []
        self.data_HR = []
        self.data_root_dir = data_root_dir
        self.LR_root_dir = LR_root_dir
        self.HR_root_dir = HR_root_dir
        self.LR_files = os.listdir(data_root_dir+"/"+LR_root_dir)
        self.HR_files = os.listdir(data_root_dir+"/"+HR_root_dir)

        #将具体文件名存入数组
        for index, name in enumerate(self.LR_files):
            #print("index:",index,"name:",name)
            self.data_LR.append(name)   #index: 0 name: 600w-3-20-7-r1.png

        for index, name in enumerate(self.HR_files):
            #print("index:",index,"name:",name)
            self.data_HR.append(name)   #index: 13 name: 600c-3-20-7-r9.png


    def __len__(self):
        return len(self.data_LR)

    def __getitem__(self, index):
        ori, label = self.data_LR[index],self.data_HR[index]
        ori_address=self.data_root_dir+"/"+self.LR_root_dir+"/"+"{}".format(ori)
        label_address = self.data_root_dir + "/" + self.HR_root_dir + "/" + "{}".format(label)

        ori_data = np.array(Image.open(ori_address))
        label_data=np.array(Image.open(label_address))
        ori_data=config.both_transforms(image=ori_data)['image']
        label_data=config.both_transforms(image=label_data)['image']
        ori_data_norm=config.highres_transform(image=ori_data)['image']
        label_data_norm=config.highres_transform(image=label_data)['image']



        return ori_data_norm,label_data_norm


def test():
    dataset = MyImageFolder(data_root_dir="data",LR_root_dir="600-wide-new",HR_root_dir="600-confocal-new")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()