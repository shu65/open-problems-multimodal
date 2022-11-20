import numpy as np


def get_selector_with_metadata_pattern(metadata, metadata_pattern):
    selector = np.ones(len(metadata), dtype=bool)
    if "donor" in metadata_pattern:
        selector &= metadata["donor"].isin(metadata_pattern["donor"])
    if "day" in metadata_pattern:
        selector &= metadata["day"].isin(metadata_pattern["day"])
    return selector
