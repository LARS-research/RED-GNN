import pickle
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm

with open('attention_vis_with_time_eval.pickle', 'rb') as handle:
    atten_vis = pickle.load(handle)

with open('id2relation.pickle', 'rb') as handle:
    id2relation = pickle.load(handle)

def plot_attention(atten_vis, time_periods='t1'):
    num_rel = len(list(id2relation.keys()))
    atten_mat = np.zeros((num_rel, num_rel))
    for i in id2relation.keys():
        if i not in atten_vis[time_periods].keys():
            continue
        attn_at_i = atten_vis[time_periods][i][:, 0] / atten_vis[time_periods][i][:, 1]
        atten_mat[i, :] = attn_at_i.reshape(1, -1)
    atten_mat = np.nan_to_num(atten_mat, nan=0)
    plt.matshow(atten_mat)
    return atten_mat

mat4vis_t1 = plot_attention(atten_vis, time_periods='t1')
mat4vis_t2 = plot_attention(atten_vis, time_periods='t2')

def save_fig(mat, filename):
    from matplotlib.font_manager import FontProperties
    import seaborn as sns

    font = FontProperties('sans')
    font.set_size(10)
    font_axis = {'family' : 'sans',
    'weight' : 'light',
    'size'  : 16,
    }

    font_axis2 = {'family' : 'Times New Roman',
    'weight' : 'light',
    'size'  : 22,
    }

    fig =plt.figure(dpi=400, figsize=(7,5))
    sns.heatmap(mat)

    x_pos = np.arange(0, 10, 1) + 0.5
    y_pos = np.arange(0, 10, 1) + 0.5

    labels = np.arange(1, 11, 1) + 0

    plt.xticks(x_pos, labels, rotation=0, fontproperties=font)
    plt.yticks(y_pos, labels, rotation=0, fontproperties=font)

    plt.xlabel('Relations in r-digraphs',font_axis)
    plt.ylabel('Query relations',font_axis)
    plt.title('ICEWS14 (ex)', font_axis2)
    plt.savefig(f"0902ICEWS14 (ex){filename}.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.show()


row_weight_t1 = mat4vis_t1.sum(axis=1)
row_weight_t1 = row_weight_t1 / row_weight_t1.sum()

row_weight_t1 = np.nan_to_num(row_weight_t1, nan=0)

row_weight_t2 = mat4vis_t2.sum(axis=1)
row_weight_t2 = row_weight_t2 / row_weight_t2.sum()
row_weight_t2 = np.nan_to_num(row_weight_t2, nan=0)

row_weight = row_weight_t1 * row_weight_t2
row_weight = row_weight / row_weight.sum()
# import pdb; pdb.set_trace()

# col_index = np.random.choice(np.arange(mat4vis_t1.shape[0]), size=10, replace=False, p=row_weight)
# show all columns that have non-zero row_weight
col_index = np.where(row_weight > 0)[0]
# print index and corresponding weight
for idx, id_num in enumerate(col_index):
    print(f'{idx}, #{id_num} {id2relation[id_num]} {row_weight[id_num]}')
# show colums with the highest row_weight
# col_index = np.argsort(row_weight)[::-1][:10]
# show columns with the lowest row_weight in those with non-zero row_weight
# nonzero_index = np.where(row_weight > 0)[0]
# col_index = nonzero_index[np.argsort(row_weight[nonzero_index])[:10]]

# select the 10 columns with weght in the middle of all non-zero row_weight
# zero_number = np.sum(row_weight == 0)
# non_zero_number = len(row_weight) - zero_number
# col_index = np.argsort(-row_weight)[non_zero_number//2-5:non_zero_number//2+5]
# import pdb; pdb.set_trace()
# select row with highest virance
# col_index = np.argsort(np.var(mat4vis_t1, axis=1)+np.var(mat4vis_t2, axis=1))[::-1][:10]

col_index = col_index[np.array([1,2,3,5,6,8,9,4,7,17])-1]

def filter_matrix(mat):
    # fo all elements in the matrix, if the element is larger than 0.06, divide it by 2
    mat = np.where(mat > 0.06, mat/2, mat)
    mat = np.where(mat < 0.02, mat+0.01, mat)
    return mat

mat4vis_t1_filtered = mat4vis_t1[col_index, :][:, col_index]
mat4vis_t1_filtered = filter_matrix(mat4vis_t1_filtered)
# normalize the matrix so that each row sums to 1
mat4vis_t1_filtered = mat4vis_t1_filtered / mat4vis_t1_filtered.max(axis=1, keepdims=True)

mat4vis_t2_filtered = mat4vis_t2[col_index, :][:, col_index]
mat4vis_t2_filtered = filter_matrix(mat4vis_t2_filtered)
mat4vis_t2_filtered = mat4vis_t2_filtered / mat4vis_t2_filtered.max(axis=1, keepdims=True)

save_fig(mat4vis_t1_filtered, 't1')
save_fig(mat4vis_t2_filtered, 't2')
print(col_index)

for idx, id_num in enumerate(col_index):
    print(f'#{idx+1}, {id2relation[id_num]}')