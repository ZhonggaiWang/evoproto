"""Teacher-known direction ranking without renormalizing full confusion rows."""
import unittest,copy
import torch
from model.directed_pair_selector import DirectedPairSelector
from model.online_directed_confusion import OnlineDirectedConfusion
from selector_legacy_reference import DirectedPairSelector as Legacy

def observer():
 o=OnlineDirectedConfusion(6,stage=2);o.updates.fill_(400)
 o.broad_image_observations.fill_(8);o.broad_pair_image_observations.fill_(3)
 o.broad_last_seen_update.fill_(399);return o

class DomainTests(unittest.TestCase):
 def test_default_matches_legacy_exactly_and_state_loads(self):
  o=observer();o.broad_ema_counts[1]=torch.tensor([200,20,30,40,50,60.])
  a=Legacy(6,stage=2);b=DirectedPairSelector(6,stage=2)
  self.assertEqual(a.get_extra_state(),b.get_extra_state())
  self.assertTrue(torch.equal(a.update(o,2400),b.update(o,2400)))
  self.assertTrue(torch.equal(a.selected_rates,b.selected_rates))
  b.load_state_dict(a.state_dict(),strict=True)
 def test_stronger_unknown_pair_cannot_hide_supported_known_pair(self):
  o=observer();o.broad_ema_counts[1]=torch.tensor([100,100,20,10,1000,0.])
  a=DirectedPairSelector(6,stage=2);b=DirectedPairSelector(6,stage=2,distillation_class_limit=4)
  self.assertEqual(int(a.update(o,2400)[1]),4)
  self.assertEqual(int(b.update(o,2400)[1]),2)
  self.assertAlmostEqual(float(b.selected_rates[1]),20/1230)
 def test_background_diagonal_unknown_sources_excluded(self):
  o=observer();o.broad_ema_counts.fill_(10)
  o.broad_ema_counts[1,0]=1000;o.broad_ema_counts[1,1]=1000
  b=DirectedPairSelector(6,stage=2,distillation_class_limit=4,min_rate=0.)
  self.assertEqual(b.update(o,2400).tolist(),[-1,2,1,1,-1,-1])
 def test_threshold_keeps_unknown_mass_in_full_denominator(self):
  o=observer();o.broad_ema_counts[1,2]=10;o.broad_ema_counts[1,4]=1990
  b=DirectedPairSelector(6,stage=2,distillation_class_limit=4)
  self.assertEqual(int(b.update(o,2400)[1]),-1)
 def test_unsupported_known_pair_does_not_fall_back_to_unknown(self):
  o=observer();o.broad_ema_counts[1,2]=20;o.broad_ema_counts[1,4]=100
  o.broad_pair_image_observations[1,2]=2
  b=DirectedPairSelector(6,stage=2,distillation_class_limit=4)
  self.assertEqual(int(b.update(o,2400)[1]),-1)
 def test_stale_known_row_withdraws_between_refreshes(self):
  o=observer();o.broad_ema_counts[1,2]=20
  b=DirectedPairSelector(6,stage=2,distillation_class_limit=4)
  b.update(o,2400);o.updates.fill_(602)
  self.assertEqual(int(b.update(o,2401)[1]),-1)
 def test_domain_metadata_serialization_and_mismatch_rejection(self):
  a=DirectedPairSelector(6,stage=2,distillation_class_limit=4)
  a.update(observer(),2400);state=copy.deepcopy(a.state_dict())
  self.assertEqual(state['_extra_state']['schema'],2)
  self.assertEqual(state['_extra_state']['distillation_class_limit'],4)
  b=DirectedPairSelector(6,stage=2,distillation_class_limit=4);b.load_state_dict(state,strict=True)
  self.assertTrue(torch.equal(a.targets,b.targets))
  with self.assertRaises(ValueError):DirectedPairSelector(6,stage=2,distillation_class_limit=3).load_state_dict(state,strict=True)
  with self.assertRaises(ValueError):DirectedPairSelector(6,stage=2).load_state_dict(state,strict=True)
 def test_invalid_domain(self):
  for limit in [True,1,7,2.5]:
   with self.subTest(limit=limit),self.assertRaises(ValueError):DirectedPairSelector(6,stage=2,distillation_class_limit=limit)

if __name__=='__main__':unittest.main()
