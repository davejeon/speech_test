import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cpu",              # was "cuda:0"
    dtype=torch.bfloat16,
    attn_implementation="sdpa",    # was "flash_attention_2"
)

# single inference
wavs, sr = model.generate_custom_voice(
    text="""
	Toyota shares soared by the most in almost four decades on Thursday after the world’s biggest carmaker raised its annual profit outlook on the back of the weak yen and strong US demand for hybrid vehicles.

    The Japanese group posted its first rise in profits in six quarters as operating income jumped 34 per cent to ¥1.13tn ($7.1bn) in the three months to September 30.

    Toyota increased its full-year profit forecast by ¥500bn to ¥3.9tn and raised its share buyback to ¥1.3tn, up from a previous plan of ¥1tn.

    Shares rallied almost 17 per cent, erasing losses for the year, building on hopes that US tariffs on car imports could be cut following the outcome of the midterm elections.

    That was the biggest one-day jump since October 1987 when markets rebounded after the Black Monday stock market crash.
    """,
    language="English", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Ryan",
    instruct="You are a professional news anchor", # Omit if not needed.
)
sf.write("output_custom_voice.wav", wavs[0], sr)

# batch inference
#wavs, sr = model.generate_custom_voice(
#    text=[
#         "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
#         "She said she would be here by noon."
#     ],
#     language=["Chinese", "English"],
#     speaker=["Vivian", "Ryan"],
#     instruct=["", "Very happy."]
# )
#sf.write("output_custom_voice_1.wav", wavs[0], sr)
#sf.write("output_custom_voice_2.wav", wavs[1], sr)
