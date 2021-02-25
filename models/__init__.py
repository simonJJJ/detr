# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#from .detr import build
#from .cgdetr import build
from .ccdetr import build
#from .deformableccdetr import build
#from .deformable_detr import build


def build_model(args):
    return build(args)
