#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""DiSMo training entrypoint.

DiSMo uses the same few-shot trainer as HyRSM. This wrapper keeps `run.py`
dispatch logic stable while reusing the maintained training implementation.
"""

from train_net_few_shot import train_few_shot
from utils.config import Config


def train_dismo(cfg):
    train_few_shot(cfg)


def main():
    cfg = Config(load=True)
    train_dismo(cfg)


if __name__ == "__main__":
    main()
