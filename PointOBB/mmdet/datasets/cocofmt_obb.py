import itertools
import logging
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from huicv.evaluation.expand_cocofmt_eval import COCOExpandEval
from huicv.evaluation.location_evaluation import LocationEvaluator
from functools import partial
import tempfile

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset
import os
from .utils import eval_rbbox_map, obb2poly_np, poly2obb_np

# add by hui, if there is not corner dataset, create one
def generate_corner_json_file_if_not_exist(ann_file, data_root, corner_kwargs):
    from huicv.corner_dataset.corner_dataset_util import generate_corner_dataset

    # generate corner json file name
    if data_root is not None:
        if not osp.isabs(ann_file):
            ann_file = osp.join(data_root, ann_file)
    origin_ann_file = ann_file
    max_tile_size, tile_overlap = corner_kwargs['max_tile_size'], corner_kwargs['tile_overlap']
    ann_file = "{}_corner_w{}h{}ow{}oh{}.json".format(
        ann_file[:-5], max_tile_size[0], max_tile_size[1], tile_overlap[0], tile_overlap[1])
    ann_dir, ann_file_name = osp.split(ann_file)
    corner_file_dir = osp.join(ann_dir, 'corner')
    ann_file = osp.join(corner_file_dir, ann_file_name)

    # generate corner dataset and save to disk, if it not exists
    if not osp.exists(ann_file):
        _ = generate_corner_dataset(origin_ann_file, save_path=ann_file, **corner_kwargs)
        print("generate corner dataset done, please re-run your code.")
        exit(0)
    return ann_file


def generate_pesudo_bbox_for_noise_data(ann_file, data_root, noise_kwargs):
    from huicv.coarse_utils.noise_data_utils import get_new_json_file_path, generate_pseudo_bbox_for_point
    # ann_file, _ = get_new_json_file_path(ann_file, data_root, 'noise', 'noisept')
    # assert osp.exists(ann_file), "{} not exist.".format(ann_file)
    ori_ann_file = ann_file
    pseudo_wh = noise_kwargs['pseudo_wh']
    if isinstance(pseudo_wh, (int, float)):
        noise_kwargs['pseudo_wh'] = pseudo_wh = (pseudo_wh, pseudo_wh)
    suffix = 'pseuw{}h{}'.format(*pseudo_wh)
    ann_file, _ = get_new_json_file_path(ori_ann_file, data_root, None, suffix)
    if not osp.exists(ann_file):
        _ = generate_pseudo_bbox_for_point(ori_ann_file, ann_file, **noise_kwargs)
        print("generate pseudo bbox for dataset done, please re-run your code.")
        exit(0)
    return ann_file


@DATASETS.register_module()
class CocoFmtObbDataset(CocoDataset):
    CLASSES = None

    def __init__(self,
                 ann_file,
                 version='le90',
                 data_root=None,
                 corner_kwargs=None,
                 train_ignore_as_bg=True,
                 noise_kwargs=None,
                 merge_after_infer_kwargs=None,
                 min_gt_size=None,
                 **kwargs):
        # add by hui, if there is not corner dataset, create one
        if corner_kwargs is not None:
            assert ann_file[-5:] == '.json', "ann_file must be a json file."
            ann_file = generate_corner_json_file_if_not_exist(ann_file, data_root, corner_kwargs)
            print("load corner dataset json file from {}".format(ann_file))
        if noise_kwargs is not None:
            if 'pseudo_wh' in noise_kwargs and noise_kwargs['pseudo_wh'] is not None:
                ann_file = generate_pesudo_bbox_for_noise_data(ann_file, data_root, noise_kwargs)
            elif 'wh_suffix' in noise_kwargs:
                from .noise_data_utils import get_new_json_file_path
                ann_file, _ = get_new_json_file_path(ann_file, data_root, noise_kwargs['sub_dir'],
                                                     noise_kwargs['wh_suffix'])
            else:
                raise ValueError('one of [pseudo_wh, wh_suffix] must be given')
            print("load noise dataset json file from {}".format(ann_file))

        self.train_ignore_as_bg = train_ignore_as_bg
        self.merge_after_infer_kwargs = merge_after_infer_kwargs

        self.min_gt_size = min_gt_size
        self.version = version

        super(CocoFmtObbDataset, self).__init__(
            ann_file,
            data_root=data_root,
            **kwargs
        )

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        if self.CLASSES is None:
            self.CLASSES = [cat['name'] for cat in self.coco.dataset['categories']]  # add by hui
        print(f'self classes:{self.CLASSES}')
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _filter_imgs(self, min_size=32):
        valid_inds = super(CocoFmtObbDataset, self)._filter_imgs(min_size)

        # filter image only contain ignore_bboxes or too small bbox
        if self.min_gt_size:
            new_valid_inds, valid_img_ids = [], []
            for i, img_id in enumerate(self.img_ids):
                valid = False
                for ann in self.coco.imgToAnns[img_id]:
                    if 'ignore' in ann and ann['ignore']:
                        continue
                    if ann['bbox'][-1] > self.min_gt_size and ann['bbox'][-2] > self.min_gt_size:
                        valid = True
                if valid:
                    new_valid_inds.append(valid_inds[i])
                    valid_img_ids.append(img_id)
            self.img_ids = valid_img_ids
            valid_inds = new_valid_inds

        print("valid image count: ", len(valid_inds))  # add by hui
        return valid_inds

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)
    
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        true_bboxes, anns_id, ann_weight = [], [], []  # add by hui,fei
        for i, ann in enumerate(ann_info):
            if self.train_ignore_as_bg and ann.get('ignore', False):  # change by hui
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                if 'true_rbox' in ann: 
                    x1,y1,x2,y2,x3,y3,x4,y4 = ann['true_rbox']
                    poly = np.array((x1,y1,x2,y2,x3,y3,x4,y4), dtype=np.float32)
                    result = poly2obb_np(poly, self.version)
                    if result is not None:
                        x, y, w, h, a = result
                        true_bboxes.append([x, y, w, h, a])
                    else:
                        print(f'poly is None: {poly}')
                        filename = img_info['file_name']
                        print(f'image info: {filename}')
                        continue
                elif 'true_bbox' in ann:
                    x1, y1, w, h = ann['true_bbox']
                    poly = np.array((x1,y1,x1+w,y1,x1+w,y1+h,x1,y1+h), dtype=np.float32)
                    result = poly2obb_np(poly, 'oc')
                    if result is not None:
                        x, y, w, h, a = result
                        true_bboxes.append([x, y, w, h, a])
                    else:
                        print(f'poly is None: {poly}')
                        filename = img_info['file_name']
                        print(f'image info: {filename}')
                        continue
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                anns_id.append(ann['id'])
                if 'ann_weight' in ann:
                    weight = ann['ann_weight']
                    ann_weight.append(weight)

        if len(true_bboxes) > 0:  # add by hui
            true_bboxes = np.array(true_bboxes, dtype=np.float32)
            anns_id = np.array(anns_id, dtype=np.int64)
            ann_weight = np.array(ann_weight, dtype=np.float32)  # add by fei

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            anns_id=anns_id,  # add by hui
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
        )
        if len(true_bboxes) > 0:  # add by hui
            ann['true_bboxes'] = true_bboxes
        if len(ann_weight) > 0: # add by fei
            ann['ann_weight'] = ann_weight
        return ann
    
    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = bboxes[i][0:5]
                    data['score'] = float(bboxes[i][5])
                    data['category_id'] = self.cat_ids[label]
                    if len(bboxes[i]) >= 7: 
                        data['ann_id'] = int(bboxes[i][6])
                    json_results.append(data)
        return json_results
    
    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files
    
    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 cocofmt_kwargs={}, skip_eval=False,
                 use_location_metric=False, location_kwargs={},
                 use_without_bbox_metric=False, without_bbox_kwargs={},
                 save_result_file=None,
                 ):  # add by hui
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        nproc=4
        nproc = min(nproc, os.cpu_count())
        iou_thr=0.5
        scale_ranges=None
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        for ann in annotations:
            ann['bboxes'] = ann['true_bboxes']
        eval_results = {}
        assert isinstance(iou_thr, float)
        mean_ap, _ = eval_rbbox_map(
            results,
            annotations,
            scale_ranges=scale_ranges,
            iou_thr=iou_thr,
            dataset=self.CLASSES,
            logger=logger,
            nproc=nproc)
        eval_results['mAP'] = mean_ap
        # else:
        #     raise NotImplementedError
        dota_eval_result = eval_results

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                import shutil
                save_result_file = './exp/latest_result.json' if save_result_file is None else save_result_file
                shutil.copy(result_files[metric], save_result_file)

            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        """
        data = super(CocoFmtObbDataset, self).__getitem__(idx)

        return data


def debug_find(data_infos, im_id=-1, filename=''):
    for idx in range(len(data_infos)):
        img_info = data_infos[idx]
        if img_info['id'] == im_id:
            return idx
        if img_info['filename'] == filename:
            return idx