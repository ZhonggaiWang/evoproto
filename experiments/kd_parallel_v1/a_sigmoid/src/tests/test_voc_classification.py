"""Check weakly supervised item loading without optional image dependencies.

Only the actual VOC12ClsDataset.__getitem__ method is loaded from its AST.
Image decoding and augmentations are substituted so this test can detect
pixel-mask access and tuple-contract regressions in the minimal torch runtime.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


def load_item_class(image_reader, pil_factory):
    source_path = Path(__file__).resolve().parents[1] / "datasets" / "voc.py"
    tree = ast.parse(source_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VOC12ClsDataset")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__getitem__"]

    class PixelMaskReadingBase:
        def __getitem__(self, index):
            raise AssertionError("Classification samples must not invoke pixel-mask loading.")

    namespace = {
        "VOC12Dataset": PixelMaskReadingBase,
        "np": np,
        "os": os,
        "imageio": SimpleNamespace(imread=image_reader),
        "Image": SimpleNamespace(fromarray=pil_factory),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["VOC12ClsDataset"]


class VOCClassificationItemTests(unittest.TestCase):
    def make_dataset(self, augmented):
        self.image = np.zeros((8, 12, 3), dtype=np.uint8)
        self.reader = Mock(return_value=self.image)
        self.pil_image = object()
        self.pil_factory = Mock(return_value=self.pil_image)
        dataset = load_item_class(self.reader, self.pil_factory)()
        dataset.img_dir = "/read-only-fixture/JPEGImages"
        dataset.name_list = np.array(["sample"])
        dataset.label_list = {"sample": np.array([1, 0, 1], dtype=np.uint8)}
        dataset.aug = augmented
        dataset._VOC12ClsDataset__transforms = Mock(return_value=("image", "local", (0, 8, 0, 12)))
        dataset.global_view2 = Mock(return_value="global2")
        return dataset

    def test_augmented_sample_reads_only_image_and_preserves_crops(self):
        dataset = self.make_dataset(augmented=True)
        name, image, labels, image_box, crops = dataset[0]
        self.reader.assert_called_once_with("/read-only-fixture/JPEGImages/sample.jpg")
        self.assertEqual(name, "sample")
        self.assertEqual(image, "image")
        self.assertIs(labels, dataset.label_list["sample"])
        self.assertEqual(image_box, (0, 8, 0, 12))
        self.assertEqual(crops, ["image", "global2", "local"])
        dataset.global_view2.assert_called_once_with(self.pil_image)
        self.assertIs(dataset._VOC12ClsDataset__transforms.call_args.kwargs["image"], self.image)

    def test_unaugmented_sample_keeps_three_value_contract(self):
        dataset = self.make_dataset(augmented=False)
        name, image, labels = dataset[0]
        self.reader.assert_called_once_with("/read-only-fixture/JPEGImages/sample.jpg")
        self.assertEqual((name, image), ("sample", "image"))
        self.assertIs(labels, dataset.label_list["sample"])
        dataset.global_view2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
