import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from predict import post_process

@pytest.fixture
def int_to_interval_map():
    return post_process(3)

def test_post_processing_function(int_to_interval_map):
    assert int_to_interval_map == {0: '0-299', 1: '300-599', 2: '600-899', 3: '900-1199'}