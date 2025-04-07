import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt
from data_utilities import *
from scipy.ndimage import label


# data directory
directory = os.path.join(os.getcwd(),   "/workspace/eli/03_test/verse809")

# load files
img_nib = nib.load(os.path.join(directory, 'verse809_CT-iso.nii.gz'))
msk_nib = nib.load(os.path.join(directory, 'verse809_CT-iso_seg.nii.gz'))
ctd_list = load_centroids(os.path.join(directory, 'verse809_CT-iso_iso-ctd.json'))

#check img zooms
zooms = img_nib.header.get_zooms()
print('img zooms = {}'.format(zooms))

#check img orientation
axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine))
print('img orientation code: {}'.format(axs_code))

# 假设 ctd_list 已经被加载
if not all(ax in ctd_list[0] for ax in axs_code):
    ctd_list.insert(0, list(axs_code))  # 将 axs_code 转换为列表并插入为第一个元素

#check centroids
print('Centroid List: {}'.format(ctd_list))

# # Resample and Reorient data
img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
msk_iso = resample_nib(msk_nib, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
ctd_iso = rescale_centroids(ctd_list, img_nib, (1,1,1))

img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
msk_iso = reorient_to(msk_iso, axcodes_to=('I', 'P', 'L'))
ctd_iso = reorient_centroids_to(ctd_iso, img_iso)

#check img zooms
zooms = img_iso.header.get_zooms()
print('img zooms = {}'.format(zooms))

#check img orientation
axs_code = nio.ornt2axcodes(nio.io_orientation(img_iso.affine))
print('img orientation code: {}'.format(axs_code))

#check centroids
print('new centroids: {}'.format(ctd_iso))

# get vocel data
im_np  = img_iso.get_fdata()
msk_np = msk_iso.get_fdata()

# get the mid-slice of the scan and mask in both sagittal and coronal planes

im_np_sag = im_np[:,:,int(im_np.shape[2]/2)]
im_np_cor = im_np[:,int(im_np.shape[1]/2),:]

msk_np_sag = msk_np[:,:,int(msk_np.shape[2]/2)]
msk_np_sag[msk_np_sag==0] = np.nan

msk_np_cor = msk_np[:,int(msk_np.shape[1]/2),:]
msk_np_cor[msk_np_cor==0] = np.nan

# 在矢状面和冠状面中间切片中分别统计标签数量
unique_labels_sag, counts_sag = np.unique(msk_np_sag[~np.isnan(msk_np_sag)], return_counts=True)
unique_labels_cor, counts_cor = np.unique(msk_np_cor[~np.isnan(msk_np_cor)], return_counts=True)

# 定义标签到脊椎段名称的映射字典
v_dict = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

def count_connected_components(slice_data, v_dict):
    label_counts = {}

    for label_value, spine_name in v_dict.items():
        # 对每个脊椎段标签生成一个二值掩码
        binary_mask = (slice_data == label_value)

        # 计算连通区域数量
        labeled_array, num_features = label(binary_mask)

        if num_features > 0:
            label_counts[spine_name] = num_features
            #label_counts[spine_name] = 1

    return label_counts


# 计算矢状面中间切片的连通区域数量
sag_counts = count_connected_components(msk_np_sag, v_dict)
print("\n脊椎段连通区域统计（矢状面中间切片）：")
for spine_name, count in sag_counts.items():
    print(f"{spine_name}   {count}")

# 计算冠状面中间切片的连通区域数量
cor_counts = count_connected_components(msk_np_cor, v_dict)
print("\n脊椎段连通区域统计（冠状面中间切片）：")
for spine_name, count in cor_counts.items():
    print(f"{spine_name}   {count}")


# plot
fig, axs = create_figure(96,im_np_sag, im_np_cor)

axs[0].imshow(im_np_sag, cmap=plt.cm.gray, norm=wdw_sbone)
axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
# plot_sag_centroids(axs[0], ctd_iso, zooms)

axs[1].imshow(im_np_cor, cmap=plt.cm.gray, norm=wdw_sbone)
axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
#plot_cor_centroids(axs[1], ctd_iso, zooms)
plt.show()

