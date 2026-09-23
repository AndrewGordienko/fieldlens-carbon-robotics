import unittest

import numpy as np

from evaluate import composite, confusion
from segmentation import HEIGHT, WIDTH, DATA, image_ids, load_scene


class PipelineTests(unittest.TestCase):
    def test_source_images_never_cross_splits(self):
        split = image_ids()
        train, val, test = map(set, (split["train"], split["val"], split["test"]))
        self.assertEqual([len(train), len(val), len(test)], [144, 40, 40])
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(len(train | val | test), 224)
        for group in range(28):
            tiles = {f"{group * 8 + offset + 1:03d}" for offset in range(8)}
            self.assertTrue(any(tiles.issubset(partition) for partition in (train, val, test)))

    def test_unlabeled_vegetation_is_ignored(self):
        if not DATA.exists():
            self.skipTest("Run prepare_data.py first")
        image, label, instances = load_scene("005")
        self.assertEqual(image.shape, (HEIGHT, WIDTH, 3))
        self.assertTrue(instances)
        self.assertGreater(int((label == 255).sum()), 0)
        self.assertTrue(set(np.unique(label)).issubset({0, 1, 2, 255}))

    def test_composite_relabels_crop_and_hides_weed(self):
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        label = np.zeros((HEIGHT, WIDTH), np.uint8)
        weed = np.zeros((HEIGHT, WIDTH), bool)
        weed[90:150, 140:200] = True
        label[weed] = 2
        crop_image = np.full((80, 80, 3), 200, np.uint8)
        crop_mask = np.ones((80, 80), bool)
        altered, altered_label, visible, overlay, actual = composite(image, label, weed, (crop_image, crop_mask), .4)
        self.assertGreater(actual, .25)
        self.assertLess(actual, .55)
        self.assertTrue(np.all(altered_label[overlay] == 1))
        self.assertFalse(np.any(visible & overlay))
        self.assertEqual(int((visible & weed).sum()), int(weed.sum()) - int((weed & overlay).sum()))
        self.assertTrue(np.all(altered[overlay] == 200))

    def test_weed_threshold_controls_prediction(self):
        probability = np.array([[[.1, .5, .4], [.1, .5, .4]]], np.float32)
        label = np.array([[2, 1]], np.uint8)
        high = confusion(probability, label, .6)
        low = confusion(probability, label, .3)
        self.assertEqual(high[2, 1], 1)
        self.assertEqual(low[2, 2], 1)
        self.assertEqual(low[1, 2], 1)


if __name__ == "__main__":
    unittest.main()
