from ss_opm.utility.metadata_utility import FEMALE_DONOR_IDS, MALE_DONOR_IDS

DONOR_METADATA_PATTERNS = [
    {
        "donor": [32606, 31800, 13176],
    },
    {
        "donor": MALE_DONOR_IDS,
    },
    {
        "donor": FEMALE_DONOR_IDS,
    },
    {
        "donor": [32606],
    },
    {
        "donor": [31800],
    },
    {
        "donor": [13176],
    },
]

DAY_METADATA_PATTERNS = [
    {
        "day": [2, 3, 4, 7],
    },
    {
        "day": [3, 4, 7],
    },
    {
        "day": [4, 7],
    },
    {
        "day": [7],
    },
]


def generate_metadata_patterns():
    day_metadata_patterns = DAY_METADATA_PATTERNS
    donor_metadata_patterns = DONOR_METADATA_PATTERNS
    metadata_patterns = []
    for donor_i in range(len(donor_metadata_patterns)):
        for day_i in range(len(day_metadata_patterns)):
            new_pattern = {}
            new_pattern.update(donor_metadata_patterns[donor_i])
            new_pattern.update(day_metadata_patterns[day_i])
            metadata_patterns.append(new_pattern)
    return metadata_patterns


def get_metadata_pattern(metadata_pattern_id):
    metadata_patterns = generate_metadata_patterns()
    return metadata_patterns[metadata_pattern_id]
