"""Real PNG checks for class-index VOC mask loading."""

import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


project = Path(__file__).resolve().parents[1]
source = project / "datasets" / "voc.py"
tree = ast.parse(source.read_text())
nodes = [node for node in tree.body if
         (isinstance(node, ast.FunctionDef) and node.name == "_read_segmentation_mask") or
         (isinstance(node, ast.ClassDef) and node.name == "VOC12Dataset")]
namespace = {"np": np, "Image": Image, "os": os, "Dataset": Dataset,
             "imageio": SimpleNamespace(imread=Mock(return_value=np.zeros((2, 4, 3), dtype=np.uint8)))}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
read_mask = namespace["_read_segmentation_mask"]
VOC12Dataset = namespace["VOC12Dataset"]


class VOCMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep generated fixtures inside the project and retain them for inspection.
        fixture_root = project / "tests" / ".runtime-tmp"
        if not fixture_root.resolve().is_relative_to(project):
            raise RuntimeError("Mask fixtures must remain inside the project.")
        fixture_root.mkdir(exist_ok=True)
        cls.fixture_dir = Path(tempfile.mkdtemp(prefix="voc-mask-", dir=fixture_root))
        cls.class_ids = np.array([[0, 1, 10, 255], [20, 2, 0, 255]], dtype=np.uint8)
        palette_image = Image.fromarray(cls.class_ids)
        palette_image.putpalette([value for index in range(256)
                                  for value in (index, (index * 3) % 256, (index * 7) % 256)])
        cls.palette_path = cls.fixture_dir / "sample.png"
        palette_image.save(cls.palette_path)
        cls.grayscale_path = cls.fixture_dir / "grayscale.png"
        Image.fromarray(cls.class_ids).save(cls.grayscale_path)
        cls.rgb_path = cls.fixture_dir / "rgb.png"
        Image.fromarray(np.stack([cls.class_ids] * 3, axis=-1)).save(cls.rgb_path)

    def test_palette_and_grayscale_preserve_ids_including_ignore(self):
        with Image.open(self.palette_path) as palette_image:
            self.assertEqual(palette_image.mode, "P")
        for path in (self.palette_path, self.grayscale_path):
            with self.subTest(path=path):
                result = read_mask(path)
                self.assertEqual(result.ndim, 2)
                np.testing.assert_array_equal(result, self.class_ids)

    def test_rgb_masks_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "2-D class-index.*mode=RGB"):
            read_mask(self.rgb_path)

    def test_training_and_validation_use_palette_ids(self):
        for stage in ("train", "val"):
            with self.subTest(stage=stage):
                dataset = VOC12Dataset.__new__(VOC12Dataset)
                dataset.stage = stage
                dataset.name_list = ["sample"]
                dataset.img_dir = "/read-only-fixture/JPEGImages"
                dataset.label_dir = str(self.fixture_dir)
                name, image, labels = dataset[0]
                self.assertEqual(name, "sample")
                self.assertEqual(image.shape, (2, 4, 3))
                np.testing.assert_array_equal(labels, self.class_ids)


if __name__ == "__main__":
    unittest.main()
