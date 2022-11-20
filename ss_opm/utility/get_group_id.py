import numpy as np


def get_group_id(metadata):
    days = [2, 3, 4, 7, 10]
    donors = [27678, 32606, 13176, 31800]
    technologies = ["citeseq", "multiome"]

    group_id = 0
    ret = np.full(len(metadata), fill_value=-1)
    for technology in technologies:
        for donor in donors:
            for day in days:
                selector = (metadata["technology"] == technology) & (metadata["donor"] == donor) & (metadata["day"] == day)
                ret[selector.values] = group_id
                group_id += 1
    assert (ret != -1).all()
    return ret
