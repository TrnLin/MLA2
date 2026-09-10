from fashion.retrieval import preprocessing as old_preprocessing
from fashion.retrieval import probe as old_probe
from fashion.retrieval import protocol as old_protocol
from fashion.task4 import preprocessing, probe, protocol


def test_old_retrieval_imports_reexport_task4_objects() -> None:
    assert old_preprocessing.PreprocessingContract is preprocessing.PreprocessingContract
    assert old_probe.extract_spatial_probe is probe.extract_spatial_probe
    assert old_protocol.build_development_views is protocol.build_development_views
