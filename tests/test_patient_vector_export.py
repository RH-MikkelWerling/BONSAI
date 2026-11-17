from types import SimpleNamespace
import pytest
import torch
from opera.run.evaluate_direct_competing_risk import patient_vector_pair


def test_outcome_attention_vectors_use_matching_shared_and_final_states(monkeypatch):
    import opera.run.export_opera_correction_arrows as arrows
    shared=torch.randn(3,4)
    final=torch.randn(3,1,4)
    def parts(model,batch,indices):
        assert indices==[2]
        return {'shared':shared,'final':final}
    monkeypatch.setattr(arrows,'representation_parts',parts)
    before,after=patient_vector_pair(SimpleNamespace(outcome_attention_readout=object()),{},2,torch.randn(3,5,4))
    torch.testing.assert_close(before,shared)
    torch.testing.assert_close(after,final[:,0])


def test_routed_vectors_retain_the_existing_target_selection():
    shared=torch.randn(3,4)
    adapted=torch.randn(3,5,4)
    def readout(x,cohort,return_weights):
        assert x is shared and return_weights
        return adapted,None
    model=SimpleNamespace(source_adapter_readout=readout)
    before,after=patient_vector_pair(model,{'cohort_fine_index':torch.zeros(3)},2,shared)
    assert before is shared
    torch.testing.assert_close(after,adapted[:,2])
    with pytest.raises(ValueError,match='checkpoint'):
        patient_vector_pair(SimpleNamespace(),{},0,shared)
