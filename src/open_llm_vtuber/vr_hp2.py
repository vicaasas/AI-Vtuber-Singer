import os,sys
parent_directory = os.path.dirname(os.path.abspath(__file__))
import logging,pdb
logger = logging.getLogger(__name__)

import librosa
import numpy as np
import soundfile as sf
import torch
from .lib.lib_v5 import nets_61968KB as Nets
from .lib.lib_v5 import spec_utils
from .lib.lib_v5.model_param_init import ModelParameters
from .lib.utils import inference

from .rvc.infer_rvc import rvc_api


class AudioPre:
    def __init__(self, agg, model_path, device, is_half, tta=False):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": tta,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("%s/lib/lib_v5/modelparams/4band_v2.json"%parent_directory)
        model = Nets.CascadedASPPNet(mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model

    def _path_audio_(
        self, music_file, ins_root=None,idx=None, vocal_root=None, format="flac", is_hp3=False
    ):
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)
        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])
        # print(bands_n)
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            (
                X_wave[d],
                _,
            ) = librosa.core.load(  # 理论上librosa读取可能对某些音频有bug，应该上ffmpeg读取，但是太麻烦了弃坑
                music_file,
                sr       = bp["sr"],
                mono     = False,
                dtype    = np.float32,
                res_type = bp["res_type"],
            )

            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            # pdb.set_trace()
            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        with torch.no_grad():
            pred, X_mag, X_phase = inference(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )

        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        input_high_end_ = spec_utils.mirroring(
            self.data["high_end_process"], y_spec_m, input_high_end, self.mp
        )
        wav_instrument = spec_utils.cmb_spectrogram_to_wave(
            y_spec_m, self.mp, input_high_end_h, input_high_end_
        )
       
        logger.info("%s instruments done" % name)
        head = "instrument"
        sf.write(
            os.path.join(
                ins_root,
                head +idx+ f".{format}",
            ),
            (np.array(wav_instrument) * 32768).astype("int16"),
            self.mp.param["sr"],
        )  #

        head = "vocal"
        input_high_end_ = spec_utils.mirroring(
            self.data["high_end_process"], v_spec_m, input_high_end, self.mp
        )
        wav_vocals = spec_utils.cmb_spectrogram_to_wave(
            v_spec_m, self.mp, input_high_end_h, input_high_end_
        )
        logger.info("%s vocals done" % name)
        vocal_path = os.path.join(
                vocal_root,
                head +idx+ f".{format}",
            )
        sf.write(
            vocal_path,
            (np.array(wav_vocals) * 32768).astype("int16"),
            self.mp.param["sr"],
        )


        # rvc
        rvc_api(dir_input=vocal_path, opt_input=vocal_path)



