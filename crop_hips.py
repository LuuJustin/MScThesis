import h5py
import numpy as np
import pandas as pd
import pydicom
import skimage
from circle_fit import circle_fit

from tqdm import tqdm

path_from_code = '../../../../tudelft.net'
partition_size = 1900


def process_dicom_file(dicom_path, pts_path):
    # set to mm/pixel, or None to disable resampling
    target_pixel_spacing = None  # 0.4

    # load image
    try:
        img = pydicom.dcmread(dicom_path)
        with open(pts_path, 'r') as f:
            # skip until start of points: line with {
            line = f.readline()
            while line and line.strip() != '{':
                line = f.readline()
            points = []
            # read points until end: line with }
            line = f.readline()
            while line and line.strip() != '}':
                points.append([float(i) for i in line.strip().split(' ')])
                line = f.readline()
        # print(f"Number of points: {len(points)}")
        points = np.array(points)

        # extract pixel spacing (mm/pixel) from the DICOM headers
        source_pixel_spacing = img.get('PixelSpacing') or img.get('ImagerPixelSpacing')
        assert source_pixel_spacing is not None, 'no pixel spacing found'
        assert source_pixel_spacing[0] == source_pixel_spacing[1], 'asymmetric pixel spacing is untested'
        pixel_spacing = source_pixel_spacing

        # resample to the required resolution
        if target_pixel_spacing is not None:
            scale_factor = source_pixel_spacing[0] / target_pixel_spacing

            img_pixels = skimage.transform.rescale(img.pixel_array, scale_factor)

            pixel_spacing = [target_pixel_spacing, target_pixel_spacing]
        else:
            img_pixels = img.pixel_array

        # are the intensities stored as MONOCHROME2 (white=max, black=min) or
        # as MONOCHROME1 (white=min, black=max)?
        photometric_interpretation = img.get('PhotometricInterpretation')
        if photometric_interpretation == 'MONOCHROME1':
            # print('Photometric interpretation MONOCHROME1: inverting intensities')
            img_pixels = np.max(img_pixels) - img_pixels
        else:
            assert photometric_interpretation == 'MONOCHROME2', \
                f'{photometric_interpretation} not supported'

        # other checks
        assert img.get('VOILUTFunction', 'LINEAR') == 'LINEAR', \
            'only supporting VOILUTFunction LINEAR'

        # define the curves: right first, then left
        SIDES = {'right': 0, 'left': 80}
        SUB_CURVES = {
            'femoral head': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            'sourcil': [70, 71, 72, 73, 74],
        }

        circles = {}
        for side, offset in SIDES.items():
            for name, curve in SUB_CURVES.items():
                xc, yc, r, sigma = circle_fit.taubinSVD(points[np.array(curve) + offset])
                circles[f'{side} {name}'] = {'xc': xc, 'yc': yc, 'r': r, 'sigma': sigma}

        combined_mask = np.zeros(shape=img_pixels.shape, dtype=np.uint8)

        LABELS = {
            'ignore': 0,
            'background': 1,
            'acetabulum': 2,
            'femur': 3,
            'joint space': 4,
        }

        # background label inside the bounding box
        combined_mask[:] = LABELS['background']

        js_bbox = {}

        for side, offset in SIDES.items():
            # define the bounding box of the segmentation region
            js_bbox[side] = bbox = {
                # top: topmost point of acetabulum curve
                'top': points[67 + offset][1],
                # medial: most medial point of the sourcil
                'medial': points[74 + offset][0],
                # lateral:
                'lateral': points[8 + offset][0],
                # bottom: medial bottom of femoral head
                'bottom': points[27 + offset][1],
            }

        # use below for hip crop instead of JSW
        hip_roi_crops = {}
        # crop to 150 mm
        roi_crop_size = int(150 / pixel_spacing[0])  # in pixels (assuming isotropic spacing for simplicity)

        for idx, side in enumerate(('right', 'left')):
            circle = circles[f'{side} femoral head']

            # Center of the femoral head in pixels
            yc_px = int(circle['yc'] / pixel_spacing[1])
            xc_px = int(circle['xc'] / pixel_spacing[0])

            # Half-size in pixels
            half_crop = roi_crop_size // 2

            # Calculate crop bounds (clamped to image size)
            y1 = max(0, yc_px - half_crop)
            y2 = min(img_pixels.shape[0], yc_px + half_crop)
            x1 = max(0, xc_px - half_crop)
            x2 = min(img_pixels.shape[1], xc_px + half_crop)

            # Slices for image and segmentation
            bbox = [slice(y1, y2), slice(x1, x2)]

            # Extract and store crops
            hip_roi_crops[side] = img_pixels[tuple(bbox)]

        return hip_roi_crops['left'], hip_roi_crops['right']
    except Exception as e:
        print(f"Error processing {dicom_path}: {str(e)}")
        return False


def resize_and_normalize(img, resize_px=224):
    # Resize the full image while preserving aspect ratio
    img_resized = skimage.transform.resize(img, (resize_px, resize_px), preserve_range=True, anti_aliasing=True)

    # Normalize to [0,1]
    img_resized = img_resized.astype(np.float32)

    # Normalize each image: mean 0, std 1
    mean = np.mean(img_resized)
    std = np.std(img_resized)
    if std > 0:
        img_resized = (img_resized - mean) / std
    else:
        img_resized = img_resized - mean  # no div by 0
    return img_resized


def save_partitioned_hip_crops(df, dataset_name, output_dir, partition_size):
    str_dt = h5py.string_dtype(encoding='utf-8')
    left_imgs, right_imgs = [], []
    left_scores, right_scores = [], []
    left_subject_ids, right_subject_ids = [], []

    part_idx = 0
    sample_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
        dicom_path = path_from_code + row['dicom']
        points_path = path_from_code + row['points']
        subject_id = str(row['subject_id'])  # Ensure it's a string

        try:
            left_crop, right_crop = process_dicom_file(dicom_path, points_path)
            left_crop = resize_and_normalize(left_crop)
            right_crop = resize_and_normalize(right_crop)
        except Exception as e:
            print(f"Skipping {dicom_path}: {e}")
            continue

        if row["Hip_OA_score_left_hip"] < 5:
            left_imgs.append(left_crop)
            left_scores.append(row["Hip_OA_score_left_hip"])
            left_subject_ids.append(subject_id)
        if row["Hip_OA_score_right_hip"] < 5:
            right_imgs.append(right_crop)
            right_scores.append(row["Hip_OA_score_right_hip"])
            right_subject_ids.append(subject_id)

        sample_count += 1

        if sample_count % partition_size == 0:
            part_filename = f"{dataset_name}_part{part_idx:02d}.h5"
            save_path = output_dir + part_filename
            with h5py.File(save_path, 'w') as grp:
                grp.create_dataset("left_hip/images", data=np.stack(left_imgs), compression="gzip")
                grp.create_dataset("left_hip/scores", data=np.array(left_scores))
                grp.create_dataset("left_hip/subject_ids", data=np.array(left_subject_ids, dtype=object), dtype=str_dt)

                grp.create_dataset("right_hip/images", data=np.stack(right_imgs), compression="gzip")
                grp.create_dataset("right_hip/scores", data=np.array(right_scores))
                grp.create_dataset("right_hip/subject_ids", data=np.array(right_subject_ids, dtype=object),
                                   dtype=str_dt)

            print(f"Saved chunk {part_idx:02d} with {len(left_imgs)} samples to {save_path}")
            part_idx += 1

            left_imgs, right_imgs = [], []
            left_scores, right_scores = [], []
            left_subject_ids, right_subject_ids = [], []

    if left_imgs:
        part_filename = f"{dataset_name}_part{part_idx:02d}.h5"
        save_path = output_dir + part_filename
        with h5py.File(save_path, 'w') as grp:
            grp.create_dataset("left_hip/images", data=np.stack(left_imgs), compression="gzip")
            grp.create_dataset("left_hip/scores", data=np.array(left_scores))
            grp.create_dataset("left_hip/subject_ids", data=np.array(left_subject_ids, dtype=object), dtype=str_dt)

            grp.create_dataset("right_hip/images", data=np.stack(right_imgs), compression="gzip")
            grp.create_dataset("right_hip/scores", data=np.array(right_scores))
            grp.create_dataset("right_hip/subject_ids", data=np.array(right_subject_ids, dtype=object), dtype=str_dt)

        print(f"Saved final chunk {part_idx:02d} with {len(left_imgs)} samples to {save_path}")


def save_hip_crops_to_hdf5(excel_path, output_dir, partition_size):
    df = pd.read_excel(excel_path)
    for dataset_name, group_df in df.groupby("dataset"):
        if dataset_name == 'OAI':
            continue
        save_partitioned_hip_crops(group_df, dataset_name, output_dir, partition_size)


save_hip_crops_to_hdf5(
    path_from_code + '/staff-umbrella/osteoarthritis2024/shared/data/CS3000_allsubjects_all_visits_files_scores.xlsx',
    path_from_code + '/staff-umbrella/MScThesisJLuu/data/', partition_size)
