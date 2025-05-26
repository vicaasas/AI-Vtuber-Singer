# python tools/uvr5/webui.py 0 False 8654 False

import os
from .vr_hp2 import AudioPre
import subprocess

def clean_path(path_str:str):
    if path_str.endswith(('\\','/')):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace('/', os.sep).replace('\\', os.sep)
    return path_str.strip(" ").strip('\'').strip("\n").strip('"').strip(" ").strip("\u202a")


parent_directory = os.path.dirname(os.path.abspath(__file__))
weight_uvr5_root = os.path.join(parent_directory, "uvr5_weights")
model_name = "HP2-人声vocals+非人声instrumentals"

device="cuda"
is_half=True

pre_fun = AudioPre(
    agg=int(10),
    model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
    device=device,
    is_half=is_half,
)

def uvr(save_root_vocal, path,idx, save_root_ins, format0):
    save_root_vocal = clean_path(save_root_vocal)
    save_root_ins = clean_path(save_root_ins)


    inp_path = os.path.join("", path)

    tmp_path = "%s/%s.reformatted.wav" % (
        os.path.join(os.environ["TEMP"]),
        os.path.basename(inp_path),
    )
    # os.system(
    #     f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
    # )
    subprocess.run(f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y', shell=True)

    inp_path = tmp_path
    pre_fun._path_audio_(
        inp_path, save_root_ins,idx, save_root_vocal, format0
    )

    # del pre_fun.model
    # del pre_fun
    # print("clean_empty_cache")
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

