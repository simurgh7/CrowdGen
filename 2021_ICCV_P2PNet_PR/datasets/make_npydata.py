import os
import numpy as np

if not os.path.exists(os.getcwd()+r'\npydata'):
    os.makedirs(os.getcwd() +r'\npydata')

'''please set your dataset path'''
shanghai_root = r'D:\crowd\data\localization\shanghaitechA'
ucf_root = r'D:\crowd\data\localization\ucf-qnrf'
jhu_root = r'D:\crowd\data\localization\jhu-crowd++'
nwpu_root = r'D:\crowd\data\localization\nwpu-crowd'

def shha_trainvaltest_npy():
    try: 
        shanghaiAtrain_path = os.path.join(shanghai_root, r'train_data\images_2048')
        shanghaiAval_path = os.path.join(shanghai_root, r'val_data\images_2048')
        shanghaiAtest_path = os.path.join(shanghai_root, r'test_data\images_2048')

        train_list = []
        for filename in os.listdir(shanghaiAtrain_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(shanghaiAtrain_path, filename))

        train_list.sort()
        np.save(os.path.join(os.getcwd(),r'npydata\shha_train.npy'), train_list)
        # print(train_list)
        val_list = []
        for filename in os.listdir(shanghaiAval_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(os.path.join(shanghaiAval_path, filename))

        val_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\shha_val.npy'), val_list)
        # print(val_list)
        test_list = []
        for filename in os.listdir(shanghaiAtest_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(shanghaiAtest_path, filename))
        test_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\shha_test.npy'), test_list)

        print("Generate ShanghaiA image list successfully")
    except:
        print("The ShanghaiA dataset path is wrong. Please check you path.")

def shhb_trainvaltest_npy():
    try: 
        shanghaiBtrain_path = os.path.join(shanghai_root, r'train_data\images_2048')
        shanghaiBval_path = os.path.join(shanghai_root, r'val_data\images_2048')
        shanghaiBtest_path = os.path.join(shanghai_root, r'test_data\images_2048')

        train_list = []
        for filename in os.listdir(shanghaiBtrain_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(shanghaiBtrain_path, filename))

        train_list.sort()
        np.save(os.path.join(os.getcwd(),r'npydata\shhb_train.npy'), train_list)

        val_list = []
        for filename in os.listdir(shanghaiBval_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(os.path.join(shanghaiBval_path, filename))

        val_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\shhb_val.npy'), val_list)

        test_list = []
        for filename in os.listdir(shanghaiBtest_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(shanghaiBtest_path, filename))
        test_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\shhb_test.npy'), test_list)

        print("generate ShanghaiA image list successfully")
    except:
        print("The ShanghaiA dataset path is wrong. Please check you path.")

def ucf_trainvaltest_npy():
    try:
        ucf_train_path = os.path.join(ucf_root, r'train_data\images_2048')
        ucf_val_path = os.path.join(ucf_root, r'val_data\images_2048')
        ucf_test_path = os.path.join(ucf_root, r'test_data\images_2048')

        train_list = []
        for filename in os.listdir(ucf_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(ucf_train_path, filename))
        train_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\ucf_train.npy'), train_list)
        print(train_list)
        val_list = []
        for filename in os.listdir(ucf_val_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(os.path.join(ucf_val_path, filename))
        val_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\ucf_val.npy'), val_list)
        print(val_list)
        test_list = []
        for filename in os.listdir(ucf_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(ucf_test_path, filename))
        test_list.sort()
        print(test_list)
        np.save(os.path.join(os.getcwd(), r'npydata\ucf_test.npy'), test_list)
        print("Generate QNRF image list successfully")
    except:
        print("The QNRF dataset path is wrong. Please check your path.")

def jhu_trainvaltest_npy():
    try:
        jhu_train_path = os.path.join(jhu_root, r'train_data\images_2048')
        jhu_val_path = os.path.join(jhu_root, r'val_data\images_2048')
        jhu_test_path = os.path.join(jhu_root, r'test_data\images_2048')

        train_list = []
        for filename in os.listdir(jhu_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(jhu_train_path, filename))
        train_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\jhu_train.npy'), train_list)

        val_list = []
        for filename in os.listdir(jhu_val_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(os.path.join(jhu_val_path, filename))
        val_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\jhu_val.npy'), val_list)

        test_list = []
        for filename in os.listdir(jhu_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(jhu_test_path, filename))
        test_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\jhu_test.npy'), test_list)

        print("Generate JHU image list successfully")
    except:
        print("The JHU dataset path is wrong. Please check your path.")

def nwpu_trainvaltest_npy():
    try:
        nwpu_train_path = os.path.join(nwpu_root, r'train_data\images_2048')
        nwpu_val_path = os.path.join(nwpu_root, r'val_data\images_2048')
        nwpu_test_path = os.path.join(nwpu_root, r'test_data\images_2048')

        train_list = []
        for filename in os.listdir(nwpu_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(nwpu_train_path, filename))
        train_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\nwpu_train.npy'), train_list)

        val_list = []
        for filename in os.listdir(nwpu_val_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(os.path.join(nwpu_val_path, filename))
        val_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\nwpu_val.npy'), val_list)

        test_list = []
        for filename in os.listdir(nwpu_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(nwpu_test_path, filename))
        test_list.sort()
        np.save(os.path.join(os.getcwd(), r'npydata\nwpu_test.npy'), test_list)
        print("Generate NWPU image list successfully")
    except:
        print("The NWPU dataset path is wrong. Please check your path.")

def adv_inference_npy(path='',name=''):
    try: 
        inference_list = []
        for filename in os.listdir(path):
            if filename.split('.')[1] == 'jpg':
                inference_list.append(os.path.join(path, filename))
                #print(os.path.join(path, filename))

        inference_list.sort()
        np.save(os.path.join(os.getcwd(),name), inference_list)

    except:
        print("The dataset path is wrong. Please check you path.")

if __name__ == '__main__':
    # shha_trainvaltest_npy()
    # ucf_trainvaltest_npy()
    # jhu_trainvaltest_npy()
    # nwpu_trainvaltest_npy()
    # SHHA
    adv_inference_npy(path=r'C:\Users\crowd\inference\shha\sasnet_unet',name=r'npydata\shha_sasnet_unet.npy')
    adv_inference_npy(path=r'C:\Users\crowd\inference\shha\p2pnet_unet',name=r'npydata\shha_p2pnet_unet.npy')
    # UCF-QNRF
    adv_inference_npy(path=r'C:\Users\crowd\inference\ucf\sasnet_unet',name=r'npydata\ucf_sasnet_unet.npy')
    adv_inference_npy(path=r'C:\Users\crowd\inference\ucf\p2pnet_unet',name=r'npydata\ucf_p2pnet_unet.npy')