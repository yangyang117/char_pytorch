import glob
import numpy as np
import os
import cv2


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (1024, 1024))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image

path_train = '/home/yyang1/data/yyang1/yyang2/Data/EyeQ-master/train'
path_test = '/home/yyang1/data/yyang1/yyang2/Data/EyeQ-master/test'

path_o = '/home/yyang2/data/yyang2/Data/EyeQ-master/vefig'

def get_image(path,path_o):
    img_list = []
    img_list = glob.glob(os.path.join(path, '*.jpeg'))
    for img in img_list:
        img_2 = load_ben_color(img)
        path_a = img.split('/')[-1]
        path_b = path.split('/')[-1]
        cv2.imwrite(path_o + '/' + path_b + '/' + path_a, img_2)


get_image(path_train, path_o)
get_image(path_test, path_o)

import os
import pandas as pd
files = os.listdir('/home/yyang2/data/yyang2/Data/EyeQ-master')
print('Label_EyeQ_train.csv' in files) #Is the labels csv in the directory?

base_image_dir = '/home/yyang2/data/yyang2/Data/EyeQ-master'
df = pd.read_csv(os.path.join(base_image_dir, 'Label_EyeQ.csv'))

df = pd.read_csv(os.path.join(base_image_dir, 'df_test_simple.csv'))
df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir, 'all', '{}'.format(x)))
df['exists'] = df['path'].map(os.path.exists) #Most of the files do not exist because this is a sample of the original dataset
df = df[df['exists']]
df = df.drop(columns=['exists'])
df = df.drop(columns=['Unnamed: 0'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)

df['DR_grade'].hist()
df.pivot_table(index='DR_grade', aggfunc=len)
df.to_csv('/home/yyang1/data/yyang1/yyang2/Data/EyeQ-master/y1_df_test.csv')



from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1)

def balance_data(class_size,df):
    train_df = df.groupby(['DR_grade']).apply(lambda x: x.sample(class_size, replace = True)).reset_index(drop = True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    print('New Data Size:', train_df.shape[0], 'Old Size:', df.shape[0])
    train_df['DR_grade'].hist(figsize = (10, 5))
    return train_df

train_df = balance_data(train_df.pivot_table(index='DR_grade', aggfunc=len).max().max(),df) # I will oversample such that all classes have the same number of images as the maximum
train_df['DR_grade'].hist(figsize=(10, 5))

train_df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/y1_df_train_over.csv')
val_df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/y1_df_test.csv')

df_train_over = pd.DataFrame(train_df)
df_train_over.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/df_train_ss.csv')

import torchvision
model = torchvision.models.densenet121()

testdf = df.loc[df.groupby(['DR_grade']).groups[0][:2644]]
testdf = testdf.apped(df.loc[df.groupby(['DR_grade']).groups[1][:]])

testdf = pd.DataFrame(testdf)

for i in range(20):
    train_df = train_df.append(df.loc[df.groupby(['DR_grade']).groups[3][:]])


train_df = pd.DataFrame(train_df)

# -----------------------------------------new-----------------------------------------

import os
import pandas as pd
files = os.listdir('D:\\data\\EyeQdata')
print('caisipictures.xls' in files) #Is the labels csv in the directory?

base_image_dir = 'D:\\data\\EyeQdata\\caisi_crop'
df = pd.read_excel(os.path.join(base_image_dir, 'caisipictures.xls'))

data_root = 'D:\\data\\EyeQdata\\caisi_crop\\'
image_file = pd.read_excel(data_root+'caisipictures.xls')
image_list = []
image_quality = []
for i in range(len(image_file)):
    image_name = str(image_file['PID'][i]) + '-' + str(image_file['StudyDate'][i]) + '-_' + str(image_file['file'][i]).split('\\')[-1][:-3] + 'png'
    image_list.append(image_name)
    image_quality.append(image_file['quality'][i])


df = pd.DataFrame({'image': image_list, 'quality': image_quality})

df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir, '{}'.format(x)))
df['exists'] = df['path'].map(os.path.exists) #Most of the files do not exist because this is a sample of the original dataset
df = df[df['exists']]
df = df.drop(columns=['exists'])
df = df.drop(columns=['Unnamed: 0'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)

df['quality'].hist()
df.pivot_table(index='quality', aggfunc=len)
df.to_csv('D:\\data\\EyeQdata\\caisi_crop\\all.csv')



from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.3)
val_df.pivot_table(index='quality', aggfunc=len)
train_df.to_csv('D:\\data\\EyeQdata\\caisi_crop\\Label_EyeQ_train.csv')
val_df.to_csv('D:\\data\\EyeQdata\\caisi_crop\\Label_EyeQ_val.csv')

def balance_data(class_size,df):
    train_df = df.groupby(['DR_grade']).apply(lambda x: x.sample(class_size, replace = True)).reset_index(drop = True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    print('New Data Size:', train_df.shape[0], 'Old Size:', df.shape[0])
    train_df['DR_grade'].hist(figsize = (10, 5))
    return train_df

train_df = balance_data(df.pivot_table(index='DR_grade', aggfunc=len).max().max(),df) # I will oversample such that all classes have the same number of images as the maximum
train_df['DR_grade'].hist(figsize=(10, 5))

train_df.to_csv('/home/yyang1/data/yyang1/yyang2/Data/EyeQ-master/y1_df_train_over.csv')


df_train_over = pd.DataFrame(train_df)
df_train_over.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/df_train_ss.csv')

import torchvision
model = torchvision.models.densenet121()

testdf = df.loc[df.groupby(['DR_grade']).groups[0][:2644]]
testdf = testdf.apped(df.loc[df.groupby(['DR_grade']).groups[1][:]])

testdf = pd.DataFrame(testdf)

for i in range(20):
    train_df = train_df.append(df.loc[df.groupby(['DR_grade']).groups[3][:]])


train_df = pd.DataFrame(train_df)

# ===============================新数据加入=================================
import glob
import pandas as pd
image_list = glob.glob(os.path.join('/home/yyang2/data/yyang2/Data/EyeQ-master/new_pictures/crops', '*.png'))
name_list = []
quality_list = []
kind_list = []
for image in image_list:
    image_name = image.split('/')[-1]
    name_list.append(image_name)
    quality = int(image_name.split('_')[1]) - 1
    quality_list.append(quality)
    kind = image_name.split('_')[0]
    kind_list.append(kind)



df = pd.DataFrame({'image': name_list, 'quality': quality_list, 'kind':kind_list})
base_image_dir = '/home/yyang2/data/yyang2/Data/EyeQ-master/new_pictures/crops'
df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir, '{}'.format(x)))

df['exists'] = df['path'].map(os.path.exists) #Most of the files do not exist because this is a sample of the original dataset
df = df[df['exists']]
df = df.drop(columns=['exists'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)
df.pivot_table(index='quality', aggfunc=len)
df.pivot_table(index=['kind', 'quality'], aggfunc=len)
df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/new_pictures/all.csv')

df = pd.read_csv('/home/yyang2/data/yyang2/Data/交大NLP资料/data_all.csv')
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.3)

test_df, vall_df = train_test_split(val_df, test_size=0.5)

train_df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/new_pictures/Label_EyeQ_train.csv')
val_df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/new_pictures/Label_EyeQ_val.csv')


def balance_data(class_size,df):
    train_df = df.groupby(['label_id']).apply(lambda x: x.sample(class_size, replace = True)).reset_index(drop = True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    print('New Data Size:', train_df.shape[0], 'Old Size:', df.shape[0])
    train_df['label_id'].hist(figsize = (10, 5))
    return train_df

train_df = balance_data(df.pivot_table(index='label_id', aggfunc=len).max().max(),df) # I will oversample such that all classes have the same number of images as the maximum
train_df['DR_grade'].hist(figsize=(10, 5))

train_df.to_csv('/home/yyang2/data/yyang2/Data/交大NLP资料/data_train_over.csv')
test_df.to_csv('/home/yyang2/data/yyang2/Data/交大NLP资料/data_test_.csv')
vall_df.to_csv('/home/yyang2/data/yyang2/Data/交大NLP资料/data_val_.csv')





