#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""DiSMo testing entrypoint.

DiSMo uses the same few-shot evaluator as HyRSM. This wrapper keeps `run.py`
dispatch logic stable while reusing the maintained test implementation.
"""

from test_net_few_shot import test_few_shot
from utils.config import Config


def test_dismo(cfg):
    test_few_shot(cfg)


def main():
    cfg = Config(load=True)
    test_dismo(cfg)


if __name__ == "__main__":
    main()
