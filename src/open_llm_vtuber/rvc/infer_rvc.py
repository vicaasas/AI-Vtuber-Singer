import numpy as np
import os
import traceback
import torch
from fairseq import checkpoint_utils
import soundfile as sf

from .lib.infer_pack.models import (
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono
)

from .my_utils import load_audio
from .vc_infer_pipeline import VC

# import sys
# now_dir = os.getcwd()
# sys.path.append(now_dir)

# 獲取 .py 檔案所在的目錄（即 A\B）
# script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
# 將工作目錄切換到 A\B
# os.chdir(script_dir)

global DoFormant, Quefrency, Timbre
Quefrency = 8.0
Timbre = 1.2
DoFormant = False


# import sqlite3
# conn = sqlite3.connect('TEMP/db:cachedb?mode=memory&cache=shared', check_same_thread=False)
# cursor = conn.cursor()

# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS formant_data (
#         Quefrency FLOAT,
#         Timbre FLOAT,
#         DoFormant INTEGER
#     )
# """)

# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS stop_train (
#         stop BOOL
#     )
# """)

# try:
#     cursor.execute("SELECT Quefrency, Timbre, DoFormant FROM formant_data")
#     row = cursor.fetchone()
#     if row is not None:
#         Quefrency, Timbre, DoFormant = row
#     else:
#         raise ValueError("No data")
    
# except (ValueError, TypeError):
#     Quefrency = 8.0
#     Timbre = 1.2
#     DoFormant = False
#     cursor.execute("DELETE FROM formant_data")
#     cursor.execute("DELETE FROM stop_train")
#     cursor.execute("INSERT INTO formant_data (Quefrency, Timbre, DoFormant) VALUES (?, ?, ?)", (Quefrency, Timbre, 0))
#     conn.commit()


# from config import Config
# config = Config()
"""
Config
"""
# class Config:
#     def __init__(self):
#         self.device = "cuda"
#         self.is_half = True
from .config import Config
config = Config()

hubert_model = None

def load_hubert():
    global hubert_model
    # hubert_base.pt 需要去額外下載
    hubert_path = os.path.join(current_dir, "hubert_base.pt")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = f"{current_dir}/weights"
sid = "mi-test_e20_s3200.pth" # model path
spk_item = 0 # 說話的人id
dir_input = 'vocal_sep' # 需要處理的音訊檔案資料夾路徑
opt_input = 'vocal_opt' # 輸出的音訊檔案資料夾路徑
inputs = [] # 直接指定的音訊檔案路徑
vc_transform1 = 0 # 音高變調
f0method1 = "rmvpe"
file_index3 = "index/added_IVF3415_Flat_nprobe_1_mi-test_v2.index"
file_index4 = "index/added_IVF3415_Flat_nprobe_1_mi-test_v2.index"
index_rate2 = 0.75 # 特徵檢索跟模型預測latent variable Z的比例 (index_feat*index_rate)
filter_radius1 = 3 # 滤波半徑，只有用harvest會用到
resample_sr1 = 0 # 後處理，需不需要resample，0代表不需要(0~48000)
rms_mix_rate1 = 1 # src跟pred音量融合比例 (src:0~pred:1)
protect1 = 0.33 # 防止電音撕裂音，越低越保留原始音訊
format1 = "wav" # 輸出格式
crepe_hop_length = 120

def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    person_path = f"{weight_root}/{sid}"
    print(f"loading {person_path}")
    cpt = torch.load(person_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk

    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    # 選擇對應的模型架構
    net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half) if if_f0 == 1 else SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    # encoder的部分inference 階段不需要
    del net_g.enc_q

    # 載入模型權重
    print(net_g.load_state_dict(cpt["weight"], strict=False))

    # 設定推理模式與精度
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    # 初始化 VC
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )

def vc_single(
    sid,
    input_audio_path0,
    input_audio_path1,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path0 is None or input_audio_path0 is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key) #變調
    try:
        if input_audio_path0 == '':
            audio = load_audio(input_audio_path1, 16000, DoFormant, Quefrency, Timbre)
            
        else:
            audio = load_audio(input_audio_path0, 16000, DoFormant, Quefrency, Timbre)
            
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1: #如果說最大值還是大於1，強制正規化到-1~1
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path1,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        print("audio output shape", audio_opt.shape)
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    format1,
    crepe_hop_length,
):
    print("!!!!")
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                None,
                f0_up_key,
                None,
                f0_method,
                file_index,
                file_index2,
                # file_big_npy,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                crepe_hop_length
            )
            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    if format1 in ["wav", "flac", "mp3", "ogg", "aac"]:
                        sf.write(
                            "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                            audio_opt,
                            tgt_sr,
                        )
                    else:
                        path = "%s/%s.wav" % (opt_root, os.path.basename(path))
                        sf.write(
                            path,
                            audio_opt,
                            tgt_sr,
                        )
                        if os.path.exists(path):
                            os.system(
                                "ffmpeg -i %s -vn %s -q:a 2 -y"
                                % (path, path[:-4] + ".%s" % format1)
                            )
                except:
                    info += traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()

def rvc_api(dir_input="vocal_sep", opt_input="vocal_opt"):
    dir_input = dir_input
    opt_input = opt_input
    vc_data = get_vc(sid, protect1, 0.33)
    vc_output3 = vc_multi(  spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                            crepe_hop_length,
                        )
    for i in vc_output3:
        print(i)

# from pydub import AudioSegment
from pydub.playback import play

if __name__ == "__main__":
    vc_data = get_vc(sid, protect1, 0.33)
    vc_output3 = vc_multi(  spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                            crepe_hop_length,
                        )
    for i in vc_output3:
        print(i)
    # print(vc_output3)
    # play(vc_output3)