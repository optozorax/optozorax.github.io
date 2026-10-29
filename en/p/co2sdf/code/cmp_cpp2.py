import subprocess, os, numpy as np
from PIL import Image
env = dict(os.environ, OMP_NUM_THREADS="32")
import render_hq
W, H, spp, steps = 180, 120, 2, 1500
for name in ["gate_N3", "neck6d", "tubes3"]:
    subprocess.run(["./trace3d", name, str(W), str(H), str(spp), str(steps), "cmp2_%s.rgbf" % name], check=True, capture_output=True, env=env)
    a = np.fromfile("cmp2_%s.rgbf" % name, dtype=np.float32).reshape(H, W, 3)
    render_hq.view(name, W, H, spp, steps, out="cmp2_py_%s.png" % name)
    b = np.asarray(Image.open("cmp2_py_%s.png" % name), dtype=np.float32) / 255
    print("%-8s C++ vs numpy (new step): mean|diff| %.4f, pixels off by >0.15: %.3f" % (name, np.abs(a - b).mean(), (np.abs(a - b).max(-1) > 0.15).mean()), flush=True)
