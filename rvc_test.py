from src.open_llm_vtuber.rvc.infer_rvc import rvc_api_single
input_dir = r"C:\Users\victo\Desktop\AI\GPT-SoVITS\output\uvr5_opt\vocal.wav"
output_dir = r"C:\Users\victo\Desktop\AI\GPT-SoVITS\output\uvr5_opt\vocal_rvc.wav"
rvc_api_single(input_pth=input_dir, output_pth=output_dir)

