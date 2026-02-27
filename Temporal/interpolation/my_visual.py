import pickle
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm

with open('attention_vis_with_time0821-2.pickle', 'rb') as handle:
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
    plt.colorbar()
    return atten_mat

mat4vis_t1 = plot_attention(atten_vis, time_periods='t1')
mat4vis_t2 = plot_attention(atten_vis, time_periods='t2')

def save_fig(mat, filename):
    from matplotlib.font_manager import FontProperties
    import seaborn as sns

    font = FontProperties('sans')
    font.set_size(10)
    # font.set_size(5)
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

    # x_pos = np.arange(0, 49, 1) + 0.5
    # y_pos = np.arange(0, 49, 1) + 0.5

    # labels = np.arange(1, 50, 1) + 0


    x_pos = np.arange(0, 10, 1) + 0.5
    y_pos = np.arange(0, 10, 1) + 0.5

    labels = np.arange(1, 11, 1) + 0

    plt.xticks(x_pos, labels, rotation=0, fontproperties=font)
    plt.yticks(y_pos, labels, rotation=0, fontproperties=font)

    plt.xlabel('Relations in r-digraphs',font_axis)
    plt.ylabel('Query relations',font_axis)
    plt.title('ICEWS14 (in)', font_axis2)
    plt.savefig(f"(new)0821 ICEWS14 (in){filename}.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.show()

row_weight_t1 = mat4vis_t1.sum(axis=1)
row_weight_t2 = mat4vis_t2.sum(axis=1)

row_weight = row_weight_t1 * row_weight_t2
row_weight = row_weight / row_weight.sum()

row_weight_t2 = mat4vis_t2.sum(axis=1)
row_weight_t2 = row_weight_t2 / row_weight_t2.sum()
row_weight_t2 = np.nan_to_num(row_weight_t2, nan=0)

row_weight = row_weight_t1 * row_weight_t2
row_weight = row_weight / row_weight.sum()

col_index = np.where(row_weight > 0)[0]

col_index = col_index[np.array([2,6,8,12,13,14,15,21,31,39])-1]
# col_index = np.random.choice(np.arange(mat4vis_t1.shape[0]), size=10, replace=False, p=row_weight)

for idx, id_num in enumerate(col_index):
    print(f'{idx}, #{id_num} {id2relation[id_num]} {row_weight[id_num]}')

mat4vis_t1_filtered = mat4vis_t1[col_index, :][:, col_index]
# mat4vis_t1_filtered = mat4vis_t1_filtered / mat4vis_t1_filtered.max(axis=1, keepdims=True)

mat4vis_t2_filtered = mat4vis_t2[col_index, :][:, col_index]
# mat4vis_t2_filtered = mat4vis_t2_filtered / mat4vis_t2_filtered.max(axis=1, keepdims=True)

save_fig(mat4vis_t1_filtered, 't1')
save_fig(mat4vis_t2_filtered, 't2')


print(col_index)

# for idx, id_num in enumerate(col_index):
#     print(f'\#{idx+1} & {id2relation[id_num]} \t \\\')