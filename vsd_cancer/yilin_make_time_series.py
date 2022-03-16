import numpy as np
import tifffile


from cellpose import models

from pathlib import Path

import cancer_functions as canf

file = Path(
    "/rds/general/user/peq10/home/firefly_link/cancer/cancer20220309/slip1/area1/long_acq/cancer20220309_slip1_area1_long_acq_blue_0.112_green_0.0673_L453_1/cancer20220309_slip1_area1_long_acq_blue_0.112_green_0.0673_L453_1_MMStack_Default.ome.tif"
)


stack = tifffile.imread(file)

ratio_stack = canf.stack2rat(stack, blue=0)


im = np.mean(stack[:100:2], 0)

model = models.Cellpose(gpu=False, model_type="cyto")
masks, flows, styles, diams = model.eval([im], diameter=30, channels=[0, 0])


tcs = tc = [canf.t_course_from_roi(ratio_stack, mask) for mask in masks]
