"""Make nurse outfit variants for the crowd test: recolour only the blue scrubs pixels of the Fuel 'Scrubs' texture (skin, hair, shoes untouched).
python scripts/follow/make_outfits.py   -> configs/follow/models/Scrubs<Colour>/{model.sdf, model.config, meshes/*}  (static, mesh URIs absolute under /x30)
Spawned in the simulator by person_mover (crowd file) with <uri>file:///x30/configs/follow/models/Scrubs<Colour></uri>.
The renderer caches textures and materials BY NAME, so every variant gets its own texture, material and mtl file name (otherwise all look navy)."""
import os
import shutil

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "external", "fuel", "models", "Scrubs", "meshes")
# colour -> (target hue 0..255 in PIL HSV, saturation scale, value scale)
OUTFITS = {"Red": (0, 1.1, 1.0), "Green": (85, 1.0, 0.95), "Purple": (200, 1.0, 1.0), "Yellow": (32, 1.2, 1.5)}


def read(p):
    return open(p, encoding="utf-8").read()


def write(p, text):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


for name, (hue, sat, val) in OUTFITS.items():
    dst = os.path.join(ROOT, "configs", "follow", "models", "Scrubs" + name)
    os.makedirs(os.path.join(dst, "meshes"), exist_ok=True)
    tex, mat, mtl = f"Scrubs{name}Diffuse.png", f"MatScrubs{name}", f"scrubs_{name.lower()}.mtl"
    for f in os.listdir(SRC):
        if f not in ("ScrubsFemaleDiffuse.png", "scrubs.mtl", "scrubs.obj"):
            shutil.copy(os.path.join(SRC, f), os.path.join(dst, "meshes", f))            # collision mesh etc.
    write(os.path.join(dst, "meshes", "scrubs.obj"), read(os.path.join(SRC, "scrubs.obj")).replace("scrubs.mtl", mtl).replace("Material.003", mat))
    write(os.path.join(dst, "meshes", mtl), read(os.path.join(SRC, "scrubs.mtl")).replace("Material.003", mat).replace("ScrubsFemaleDiffuse.png", tex))
    hsv = np.array(Image.open(os.path.join(SRC, "ScrubsFemaleDiffuse.png")).convert("RGB").convert("HSV")).astype(np.float32)
    blue = (hsv[..., 0] > 130) & (hsv[..., 0] < 190) & (hsv[..., 1] > 60)                 # navy scrubs only
    hsv[..., 0][blue] = hue
    hsv[..., 1][blue] = np.clip(hsv[..., 1][blue] * sat, 0, 255)
    hsv[..., 2][blue] = np.clip(hsv[..., 2][blue] * val, 0, 255)
    Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB").save(os.path.join(dst, "meshes", tex))
    base = f"file:///x30/configs/follow/models/Scrubs{name}/meshes"
    write(os.path.join(dst, "model.sdf"), f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="Scrubs{name}">
    <static>true</static>
    <link name="body">
      <visual name="visual"><geometry><mesh><uri>{base}/scrubs.obj</uri></mesh></geometry></visual>
      <collision name="collision"><geometry><mesh><uri>{base}/Scrubs_Col.obj</uri></mesh></geometry></collision>
    </link>
  </model>
</sdf>
""")
    write(os.path.join(dst, "model.config"), f"""<?xml version="1.0"?>
<model><name>Scrubs{name}</name><version>1.0</version><sdf version="1.6">model.sdf</sdf>
<description>Fuel 'Scrubs' with recoloured scrubs ({name}) for the follow_person crowd test.</description></model>
""")
    print("made", dst, "blue pixels:", int(blue.sum()))
