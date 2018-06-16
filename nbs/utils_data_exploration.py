
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import re
import glob


## Extract image ids
add_prefix = lambda x: '../data/' + x

train_images = glob.glob(add_prefix('train_photos/*.jpg'))
test_images = glob.glob(add_prefix('test_photos/*.jpg'))


# function to extract ids
extract_id = lambda x: re.match('.*/([0-9]+).jpg', x).group(1)

train_ids = list(map(extract_id, train_images))
test_ids = list(map(extract_id, test_images))

df_train_labels = pd.read_csv(add_prefix('train.csv')).dropna()
df_train_labels['labels'] = df_train_labels['labels'].apply(lambda x: tuple(sorted(int(l) for l in x.split())))
df_train_labels.set_index('business_id', inplace=True)


df_train_photo_to_biz_ids = pd.read_csv(add_prefix('train_photo_to_biz_ids.csv'))
photos_in_train_biz = df_train_photo_to_biz_ids.groupby('business_id')['photo_id'].apply(list)
df_train_labels['photos'] = photos_in_train_biz
df_train_labels['n_photo'] = df_train_labels.photos.apply(len)


df_test_photo_to_biz_ids = pd.read_csv(add_prefix('test_photo_to_biz.csv'))


df_test_photo_to_biz_ids.head()
photos_in_test_biz = df_test_photo_to_biz_ids.groupby('business_id')['photo_id'].apply(list).to_dict()
test_biz_ids = list(photos_in_test_biz.keys())


def photos_in_biz(i):
    return df_train_labels.loc[i].photos


label_desc_dict = dict(map(lambda x: (int(x[0]), x[1]),
                           map(lambda x: x.split(': '), '''0: good_for_lunch
1: good_for_dinner
2: takes_reservations
3: outdoor_seating
4: restaurant_is_expensive
5: has_alcohol
6: has_table_service
7: ambience_is_classy
8: good_for_kids
'''.splitlines())))


def encode_label(l):
    res = np.zeros(len(label_desc_dict))
    for i in l:
        res[i] = 1
    return res

def decode_label(x):
    return tuple(np.where(x==1)[0])


# ## photo display


from types import ModuleType


def get_image(id, test=False):
    if test:
        return add_prefix('test_photos/{}.jpg'.format(id))
    else:
        return add_prefix('train_photos/class0/{}.jpg'.format(id))


def show_image(id, test=False, ax=plt, msg=None):
    ax.imshow(plt.imread(get_image(id, test)), extent=[0,100,0,100])
    ax.grid(False)
    title = str(id)
    ax.axis('off')
    
    #if test:
    #    title = 'test '+title
    #if msg is not None:
    #    title += ': ' + str(msg)
    #if isinstance(ax, ModuleType):
    #    ax.title(title)
    #else:
    #    ax.set_title(title)


def show_photos(photos, m, n, msgs=None):
    with_test = isinstance(photos[0], tuple)
    if msgs is None:
        msgs = [None] * len(photos)
    
    fig, axes = plt.subplots(m, n, figsize=(15,10))
    for ax, i, msg in zip(axes.ravel(), photos, msgs):
        if with_test:
            show_image(i[0], i[1], ax, msg)
        else:
            show_image(i, ax=ax, msg=msg)
    return fig


def show_photos_in_bussiness(biz_id, m=3, n=3, seed=42):
    max_photo_size = m*n
    photos = df_train_labels.loc[biz_id].photos
    if len(photos) <= max_photo_size:
        sample_images = photos
    else:
        rng = np.random.RandomState(seed)
        sample_images = rng.choice(photos, max_photo_size, replace=False)
    for l in df_train_labels.loc[biz_id]['labels']:
        print('{}/{} in buisness {}'.format(len(sample_images), len(photos), biz_id))
    fig = show_photos(sample_images, m, n)
    fig.suptitle('{}/{} in business {}'.format(len(sample_images), len(photos), biz_id))



def write_submission(L, fname, biz=test_biz_ids):
    with open(fname, 'w') as f:
        f.write('business_id, labels\n')
        for i, l in zip(biz, L):
            f.write('{}, {}\n'.format(i, ' '.join(map(str, np.where(l==1)[0]))))


# show_photos_in_bussiness(1000, seed=4)

