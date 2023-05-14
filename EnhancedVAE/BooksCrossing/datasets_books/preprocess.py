import os
import shutil
import sys

import numpy as np
from scipy import sparse
import pandas as pd

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=1, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users. 
    if min_sc > 0:
        itemcount = get_count(tp, 'Anime_ID')
        tp = tp[tp['Anime_ID'].isin(itemcount.loc[itemcount['size'] >= min_sc].index)]
    
    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'User_ID')
        tp = tp[tp['User_ID'].isin(usercount.loc[usercount['size'] >= min_uc].index)]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'User_ID'), get_count(tp, 'Anime_ID') 
    return tp, usercount, itemcount

### change `DATA_DIR` to the location where movielens-20m dataset sits

#DATA_DIR = '/content/drive/MyDrive/anime'
data = pd.read_csv(os.path.join('book_history.dat'), sep = '\t')

# for i in range(len(data)):
#   if(data['Feedback'][i] <= 2):
#     data['Feedback'][i] = 1
#   elif(data['Feedback'][i] <= 4):
#     data['Feedback'][i] = 2
#   elif(data['Feedback'][i] <= 6):
#     data['Feedback'][i] = 3
#   elif(data['Feedback'][i] <= 8):
#     data['Feedback'][i] = 4
#   elif(data['Feedback'][i] <= 10):
#     data['Feedback'][i] = 5
  
# binarize the data (only keep Feedback >= 3.5)
raw_data = data
# raw_data = raw_data[raw_data['Feedback'] > 3.5]
# raw_data.describe()

raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=0, min_sc=0)
sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("After filtering, there are %d watching events from %d users and %d books (sparsity: %.3f%%)" % 
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

unique_uid = user_activity.index

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]

n_users = unique_uid.size
n_heldout_users = 236

tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['User_ID'].isin(tr_users)]
unique_sid = pd.unique(train_plays['Anime_ID'])
show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
pro_dir = "ML_20m"

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('User_ID')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 1:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 100 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    
    return data_tr, data_te
vad_plays = raw_data.loc[raw_data['User_ID'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['Anime_ID'].isin(unique_sid)]
vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data['User_ID'].isin(te_users)]
test_plays = test_plays.loc[test_plays['Anime_ID'].isin(unique_sid)]
test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['User_ID']))
    sid = list(map(lambda x: show2id[x], tp['Anime_ID']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])
train_data = numerize(train_plays)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
test_data_tr = numerize(test_plays_tr)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
test_data_te = numerize(test_plays_te)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['User_ID']))
    sid = list(map(lambda x: show2id[x], tp['Anime_ID']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

train_data = numerize(train_plays)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
