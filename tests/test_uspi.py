import unittest
import torch
from uspi_policy import StableConfig, paired_lower_bound, continuity_mask, local_trajectory, stable_select
from da_control_policy import _candidate_actions
from da_control_rule import rule_actions


def fixture():
    s=torch.zeros(1,307); s[0,3]=5.; s[0,9]=3.5
    wp=s[0,-36:].view(12,3); wp[:,0]=torch.arange(12)*2.
    # Reference and small feasible improvement on a straight lane.
    a=torch.tensor([[.75,0.,0.],[.74,0.,0.],[.75,.7,0.]])
    values=torch.zeros(3,3,4)
    values[:,1:,:]=torch.tensor([1.,1.,-1.,-1.])
    return s,a,values,torch.full((3,),10),torch.ones(4)


class TestUSPI(unittest.TestCase):
    def test_paired_bound_is_stricter_than_worst_head(self):
        v=torch.tensor([[[2.,3.,4.,5.],[3.,5.,3.,4.]],[[10.,20.,30.,40.],[13.,24.,27.,37.]]])
        b=paired_lower_bound(v,torch.ones(4),1.)
        torch.testing.assert_close(b[0],torch.zeros(4))
        torch.testing.assert_close(b[1],torch.tensor([0.,1.,0.,0.]))
    def test_candidate_is_selected_and_large_steer_rejected(self):
        s,a,v,n,sc=fixture()
        i,log=stable_select(s,a,v,n,sc,a[0],StableConfig())
        self.assertEqual(i,1); self.assertEqual(log['gate_trajectory_count'],1)
    def test_jerk_degradation_rejected(self):
        s,a,v,n,sc=fixture();v[:,1,3]=1.
        i,_=stable_select(s,a,v,n,sc,a[0],StableConfig());self.assertEqual(i,0)
    def test_unsupported_candidate_rejected(self):
        s,a,v,n,sc=fixture();n[1]=4
        i,_=stable_select(s,a,v,n,sc,a[0],StableConfig());self.assertEqual(i,0)
    def test_missing_route_falls_back(self):
        s,a,v,n,sc=fixture();s[0,-36:]=0
        i,_=stable_select(s,a,v,n,sc,a[0],StableConfig());self.assertEqual(i,0)
    def test_nonfinite_reference_falls_back(self):
        s,a,v,n,sc=fixture();v[:,0,0]=float('nan')
        i,_=stable_select(s,a,v,n,sc,a[0],StableConfig());self.assertEqual(i,0)
    def test_emergency_brake_never_weakened(self):
        s,a,v,n,sc=fixture();s[0,7]=4.;a[0]=rule_actions(s)[0];a[1]=torch.tensor([0.,0.,.58])
        i,_=stable_select(s,a,v,n,sc,a[0],StableConfig());self.assertEqual(i,0)
    def test_candidate_exclusivity(self):
        s,a,_,_,_=fixture(); x,_=_candidate_actions(a[0],torch.tensor([[.4,.2,.5]]),None)
        self.assertTrue(bool(((x[:,0]*x[:,2])==0).all()))
    def test_local_rollout_measures_steering_deviation(self):
        s,a,_,_,_=fixture();d,h,ok=local_trajectory(s,a,StableConfig());self.assertTrue(bool(ok.all()));self.assertGreater(float(d[2]),float(d[0]));self.assertGreater(float(h[2]),float(h[0]))
    def test_lane_margin_rejects_nonref_but_reference_remains(self):
        s,a,v,n,sc=fixture();s[0,9]=.2
        i,_=stable_select(s,a,v,n,sc,a[0],StableConfig());self.assertEqual(i,0)

if __name__=='__main__': unittest.main()
