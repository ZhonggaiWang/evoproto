"""Keep pretrained initialization available with both timm config names."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from model.backbone import vit


class VitPretrainedCompatibilityTests(unittest.TestCase):
    def test_current_timm_receives_pretrained_cfg_and_original_options(self):
        model = SimpleNamespace(default_cfg={'url': 'test-weights', 'num_classes': 1000})
        observed = {}

        def current_loader(net, pretrained_cfg=None, **kwargs):
            observed.update(model=net, config=pretrained_cfg, options=kwargs)

        with patch.object(vit, 'timm_load_pretrained', current_loader):
            vit.load_pretrained(model, num_classes=1000, in_chans=3, filter_fn=vit._conv_filter)
        self.assertIs(observed['model'], model)
        self.assertIs(observed['config'], model.default_cfg)
        self.assertEqual(observed['options']['num_classes'], 1000)
        self.assertIs(observed['options']['filter_fn'], vit._conv_filter)

    def test_older_timm_receives_default_cfg_without_changing_it(self):
        model = SimpleNamespace(default_cfg={'url': 'fallback'})
        explicit = {'url': 'chosen-weights'}
        observed = {}

        def older_loader(net, default_cfg=None, **kwargs):
            observed.update(config=default_cfg, options=kwargs)

        with patch.object(vit, 'timm_load_pretrained', older_loader):
            vit.load_pretrained(model, default_cfg=explicit, num_classes=1000)
        self.assertIs(observed['config'], explicit)
        self.assertEqual(model.default_cfg, {'url': 'fallback'})


if __name__ == '__main__':
    unittest.main()
