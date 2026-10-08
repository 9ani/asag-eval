import re

import asag_eval


def test_version_is_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", asag_eval.__version__)
