import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging
import json

from .base_dataset import BaseDataset, BaseDatasetSplit
from .utils import BEVBox3D
from ..utils import make_dir, DATASET


log = logging.getLogger(__name__)

# Expect point clouds to be in npy format with train, val and test files in separate folders.
# Expected format of txt files : ['x', 'y', 'z', 'feat 1', 'feat_2', ........,'feat_n'].
# For test files, format should be : ['x', 'y', 'z', 'feat_1', 'feat_2', ........,'feat_n'].


class BBLightCodeSplit(BaseDatasetSplit):
    """This class is used to create a custom dataset split.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
        'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        label_path = pc_path.replace('pointclouds', 'labels').replace('.txt', '.json')    

        # Dummy camera calibration as no rotations between bounidng boxes and/or point clouds exist
        dummy_cam = np.eye(4)
        calib = {'world_cam': dummy_cam}

        point_cloud = self.dataset.read_lidar(pc_path)
        label = self.dataset.read_label(label_path, calib)

        data = {
            'point': point_cloud,
            'feat': None,
            'bounding_boxes': label,
            'calib': calib
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


class BBLightCode(BaseDataset):
    """A template for customized dataset that you can use with a dataloader to
    feed data when training a model. This inherits all functions from the base
    dataset and can be modified by users. Initialize the function by passing the
    dataset and other details.

    Args:
        dataset_path: The path to the dataset to use.
        name: The name of the dataset.
        cache_dir: The directory where the cache is stored.
        use_cache: Indicates if the dataset should be cached.
        num_points: The maximum number of points to use when splitting the dataset.
        ignored_label_inds: A list of labels that should be ignored in the dataset.
        test_result_folder: The folder where the test results should be stored.
    """

    def __init__(self,
                 dataset_path,
                 name='bb_lightcode',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 test_result_folder='./test',
                 **kwargs):

        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.dataset_path = cfg.dataset_path
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)

        # Read in point cloud files
        self.train_files = sorted(glob.glob(join(self.dataset_path, "training", "pointclouds", "*.txt")))
        self.val_files = sorted(glob.glob(join(self.dataset_path, "validation", "pointclouds", "*.txt")))
        self.test_files = sorted(glob.glob(join(self.dataset_path, "testing", "pointclouds", "*.txt")))

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'Car',
            1: 'Pedestrian',
            2: 'Bicycle',
            3: 'Motorcycle'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        return np.loadtxt(path, dtype=np.float32)

    @staticmethod
    def read_label(path, calib):
        """Reads labels from SUSTechPOINTS annotation tool format.

        Returns:
            The data objects with bound boxes information.
        """
        if not Path(path).exists():
            return []

        with open(path, "r") as f:
            data = json.load(f)

            objects = []
            for line in data:

                label_str = str(line["obj_type"])
                
                center = np.array([line["psr"]["position"]["x"], line["psr"]["position"]["y"], line["psr"]["position"]["z"]], dtype=np.float32)
                
                size = np.array([line["psr"]["scale"]["y"], line["psr"]["scale"]["z"], line["psr"]["scale"]["x"]], dtype=np.float32)

                yaw = np.array(line["psr"]["rotation"]["z"], dtype=np.float32)
                yaw -= np.pi / 2
                yaw *= -1
                while yaw < -np.pi:
                    yaw += (np.pi * 2)
                while yaw > np.pi:
                    yaw -= (np.pi * 2)
                
                objects.append(BEVBox3D(center=center, size=size, yaw=yaw, label_class=label_str, confidence=-1.0, world_cam=calib['world_cam']))

        return objects

    
    @staticmethod
    def read_calib(path):
        """Reads calibiration for the dataset. You can use them to compare
        modeled results to observed results.
        """
        pass

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return BBLightCodeSplit(self, split=split)

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
             ValueError: Indicates that the split name passed is incorrect. The
             split name should be one of 'training', 'test', 'validation', or
             'all'.
        """

        # Additional two keywords to retrieve the data with ground truth bounding boxes for testing
        if split in ['test', 'testing', 'test_with_bb', 'testing_with_bb']:
            self.rng.shuffle(self.test_files)
            return self.test_files
        elif split in ['val', 'validation']:
            self.rng.shuffle(self.val_files)
            return self.val_files
        elif split in ['train', 'training']:
            self.rng.shuffle(self.train_files)
            return self.train_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
            return files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attrs):
        """Saves the output prediction to .json file.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """

        make_dir(self.cfg.test_result_folder)
        for attr, res in zip(attrs, results):
            name = attr['name']
            path = join(self.cfg.test_result_folder, name + '.json')
            
            json_label_array = []
            for box in res:

                yaw = box.yaw + np.pi / 2
                yaw *= -1
                while yaw < -np.pi:
                    yaw += (np.pi * 2)
                while yaw > np.pi:
                    yaw -= (np.pi * 2)
                
                json_label = {
                    "obj_id": str(0),
                    "obj_type": box.label_class,
                    "psr": {
                        "position": {
                            "x": float(box.center[0]),
                            "y": float(box.center[1]),
                            "z": float(box.center[2])
                        },
                        "rotation": {
                            "x": float(0.0),
                            "y": float(0.0),
                            "z": float(yaw)
                        },
                        "scale": {
                            "x": float(box.size[2]),
                            "y": float(box.size[0]),
                            "z": float(box.size[1])
                        }
                    }
                }
                
                json_label_array.append(json_label)
            
            with open(path, "w") as outfile:
                outfile.write(json.dumps(json_label_array, indent=2))


DATASET._register_module(BBLightCode)
