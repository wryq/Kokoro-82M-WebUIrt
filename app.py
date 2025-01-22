
from KOKORO.models import build_model
from KOKORO.utils import tts,tts_file_name,podcast
import sys
sys.path.append('.')
import os 
# os.system("python download_model.py")
import torch
import gc 
import platform
import shutil
base_path=os.getcwd()
def clean_folder_before_start():
    global base_path
    folder_list=["dummy","TTS_DUB","kokoro_audio"]
    for folder in folder_list:
        if os.path.exists(f"{base_path}/{folder}"):
            try:
                shutil.rmtree(f"{base_path}/{folder}")
            except:
                pass
            os.makedirs(f"{base_path}/{folder}", exist_ok=True)
clean_folder_before_start()

print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
MODEL = build_model('./KOKORO/kokoro-v0_19.pth', device)
print("Model loaded successfully.")

def tts_maker(text,voice_name="af_bella",speed = 0.8,trim=0,pad_between=0,save_path="temp.wav",remove_silence=False,minimum_silence=50):
    # Sanitize the save_path to remove any newline characters
    save_path = save_path.replace('\n', '').replace('\r', '')
    global MODEL
    audio_path=tts(MODEL,device,text,voice_name,speed=speed,trim=trim,pad_between_segments=pad_between,output_file=save_path,remove_silence=remove_silence,minimum_silence=minimum_silence)
    return audio_path


model_list = ["kokoro-v0_19.pth", "kokoro-v0_19-half.pth"]
current_model = model_list[0]

def update_model(model_name):
    """
    Updates the TTS model only if the specified model is not already loaded.
    """
    global MODEL, current_model
    if current_model == model_name:
        return f"Model already set to {model_name}"  # No need to reload
    model_path = f"./KOKORO/{model_name}"  # Default model path
    if model_name == "kokoro-v0_19-half.pth":
        model_path = f"./KOKORO/fp16/{model_name}"  # Update path for specific model
    # print(f"Loading new model: {model_name}")
    del MODEL  # Cleanup existing model
    gc.collect()
    torch.cuda.empty_cache()  # Ensure GPU memory is cleared
    MODEL = build_model(model_path, device)
    current_model = model_name
    return f"Model updated to {model_name}"


def manage_files(file_path):
    if os.path.exists(file_path):
        file_extension = os.path.splitext(file_path)[1]  # Get file extension
        file_size = os.path.getsize(file_path)  # Get file size in bytes
        # Check if file is a valid .pt file and its size is ≤ 5 MB
        if file_extension == ".pt" and file_size <= 5 * 1024 * 1024:
            return True  # File is valid and kept
        else:
            os.remove(file_path)  # Delete invalid or oversized file
            return False
    return False  # File does not exist



def text_to_speech(text, model_name="kokoro-v0_19.pth", voice_name="af", speed=1.0, pad_between_segments=0, remove_silence=True, minimum_silence=0.20,custom_voicepack=None,trim=0.0):
    """
    Converts text to speech using the specified parameters and ensures the model is updated only if necessary.
    """
    update_status = update_model(model_name)  # Load the model only if required
    # print(update_status)  # Log model loading status
    if not minimum_silence:
        minimum_silence = 0.05
    keep_silence = int(minimum_silence * 1000)
    save_at = tts_file_name(text)
    # print(voice_name,custom_voicepack)
    if custom_voicepack:
        if manage_files(custom_voicepack):
            voice_name = custom_voicepack
        else:
            gr.Warning("Upload small size .pt file only. Using the Current voice pack instead.")
    audio_path = tts_maker(
        text, 
        voice_name, 
        speed, 
        trim, 
        pad_between_segments, 
        save_at, 
        remove_silence, 
        keep_silence
    )
    return audio_path




import gradio as gr

# voice_list = [
#     'af',  # Default voice is a 50-50 mix of af_bella & af_sarah
#     'af_bella', 'af_sarah', 'am_adam', 'am_michael',
#     'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
# ]



import os

# Get the list of voice names without file extensions
voice_list = [
    os.path.splitext(filename)[0]
    for filename in os.listdir("./KOKORO/voices")
    if filename.endswith('.pt')
]

# Sort the list based on the length of each name
voice_list = sorted(voice_list, key=len)

def toggle_autoplay(autoplay):
    return gr.Audio(interactive=False, label='Output Audio', autoplay=autoplay)

with gr.Blocks() as demo1:
    gr.Markdown("# Batched TTS")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label='Enter Text',
                lines=3,
                placeholder="Type your text here..."
            )
            with gr.Row():
                voice = gr.Dropdown(
                    voice_list, 
                    value='af', 
                    allow_custom_value=False, 
                    label='Voice', 
                    info='Starred voices are more stable'
                )
            with gr.Row():
                generate_btn = gr.Button('Generate', variant='primary')
            with gr.Accordion('Audio Settings', open=False):
                model_name=gr.Dropdown(model_list,label="Model",value=model_list[0])
                speed = gr.Slider(
                    minimum=0.25, maximum=2, value=1, step=0.1, 
                    label='⚡️Speed', info='Adjust the speaking speed'
                )
                remove_silence = gr.Checkbox(value=False, label='✂️ Remove Silence From TTS')
                minimum_silence = gr.Number(
                    label="Keep Silence Upto (In seconds)", 
                    value=0.05
                )
                
                # trim = gr.Slider(
                #     minimum=0, maximum=1, value=0, step=0.1, 
                #     label='🔪 Trim', info='How much to cut from both ends of each segment'
                # )   
                pad_between = gr.Slider(
                    minimum=0, maximum=2, value=0, step=0.1, 
                    label='🔇 Pad Between', info='Silent Duration between segments [For Large Text]'
                )
                
                custom_voicepack = gr.File(label='Upload Custom VoicePack .pt file')
                
        with gr.Column():
            audio = gr.Audio(interactive=False, label='Output Audio', autoplay=True)
            with gr.Accordion('Enable Autoplay', open=False):
                autoplay = gr.Checkbox(value=True, label='Autoplay')
                autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])

    text.submit(
        text_to_speech, 
        inputs=[text, model_name,voice, speed, pad_between, remove_silence, minimum_silence,custom_voicepack], 
        outputs=[audio]
    )
    generate_btn.click(
        text_to_speech, 
        inputs=[text,model_name, voice, speed, pad_between, remove_silence, minimum_silence,custom_voicepack], 
        outputs=[audio]
    )

def podcast_maker(text,remove_silence=False,minimum_silence=50,model_name="kokoro-v0_19.pth"):
    global MODEL,device
    update_model(model_name)
    if not minimum_silence:
        minimum_silence = 0.05
    keep_silence = int(minimum_silence * 1000)
    podcast_save_at=podcast(MODEL, device,text,remove_silence=remove_silence, minimum_silence=keep_silence)
    return podcast_save_at
    


dummpy_example="""{af} Hello, I'd like to order a sandwich please.                                                         
{af_sky} What do you mean you're out of bread?                                                                      
{af_bella} I really wanted a sandwich though...                                                              
{af_nicole} You know what, darn you and your little shop!                                                                       
{bm_george} I'll just go back home and cry now.                                                                           
{am_adam} Why me?"""
with gr.Blocks() as demo2:
    gr.Markdown(
        """
    # Multiple Speech-Type Generation
    This section allows you to generate multiple speech types or multiple people's voices. Enter your text in the format shown below, and the system will generate speech using the appropriate type. If unspecified, the model will use "af" voice.
    Format:
    {voice_name} your text here
    """
    )
    with gr.Row():
        gr.Markdown(
            """
            **Example Input:**                                                                      
            {af} Hello, I'd like to order a sandwich please.                                                         
            {af_sky} What do you mean you're out of bread?                                                                      
            {af_bella} I really wanted a sandwich though...                                                              
            {af_nicole} You know what, darn you and your little shop!                                                                       
            {bm_george} I'll just go back home and cry now.                                                                           
            {am_adam} Why me?!                                                                         
            """
        )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label='Enter Text',
                lines=7,
                placeholder=dummpy_example
            )
            with gr.Row():
                generate_btn = gr.Button('Generate', variant='primary')
            with gr.Accordion('Audio Settings', open=False):
                remove_silence = gr.Checkbox(value=False, label='✂️ Remove Silence From TTS')
                minimum_silence = gr.Number(
                    label="Keep Silence Upto (In seconds)", 
                    value=0.20
                )
        with gr.Column():
            audio = gr.Audio(interactive=False, label='Output Audio', autoplay=True)
            with gr.Accordion('Enable Autoplay', open=False):
                autoplay = gr.Checkbox(value=True, label='Autoplay')
                autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])

    text.submit(
        podcast_maker, 
        inputs=[text, remove_silence, minimum_silence], 
        outputs=[audio]
    )
    generate_btn.click(
        podcast_maker, 
        inputs=[text, remove_silence, minimum_silence], 
        outputs=[audio]
    )




import shutil
import os

# Ensure the output directory exists
output_dir = "./temp_audio"
os.makedirs(output_dir, exist_ok=True)









#@title Generate Audio File From Subtitle
# from tqdm.notebook import tqdm
from tqdm import tqdm
import subprocess
import json
import pysrt
import os
from pydub import AudioSegment
import shutil
import uuid
import re
import time

# os.chdir(install_path)

def your_tts(text,audio_path,actual_duration,speed=1.0):
  global srt_voice_name
  model_name="kokoro-v0_19.pth"
  tts_path=text_to_speech(text, model_name, voice_name=srt_voice_name,speed=speed,trim=1.0)
#   print(tts_path)
  tts_audio = AudioSegment.from_file(tts_path)
  tts_duration = len(tts_audio)
  if tts_duration > actual_duration:
    speedup_factor = tts_duration / actual_duration
    tts_path=text_to_speech(text, model_name, voice_name=srt_voice_name,speed=speedup_factor,trim=1.0)
#   print(tts_path)
  shutil.copy(tts_path,audio_path)



base_path="."
import datetime
def get_current_time():
    # Return current time as a string in the format HH_MM_AM/PM
    return datetime.datetime.now().strftime("%I_%M_%p")

def get_subtitle_Dub_path(srt_file_path,Language="en"):
  file_name = os.path.splitext(os.path.basename(srt_file_path))[0]
  if not os.path.exists(f"{base_path}/TTS_DUB"):
    os.mkdir(f"{base_path}/TTS_DUB")
  random_string = str(uuid.uuid4())[:6]
  new_path=f"{base_path}/TTS_DUB/{file_name}_{Language}_{get_current_time()}_{random_string}.wav"
  return new_path








def clean_srt(input_path):
    file_name = os.path.basename(input_path)
    output_folder = f"{base_path}/save_srt"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_path = f"{output_folder}/{file_name}"

    def clean_srt_line(text):
        bad_list = ["[", "]", "♫", "\n"]
        for i in bad_list:
            text = text.replace(i, "")
        return text.strip()

    # Load the subtitle file
    subs = pysrt.open(input_path)

    # Iterate through each subtitle and print its details
    with open(output_path, "w", encoding='utf-8') as file:
        for sub in subs:
            file.write(f"{sub.index}\n")
            file.write(f"{sub.start} --> {sub.end}\n")
            file.write(f"{clean_srt_line(sub.text)}\n")
            file.write("\n")
        file.close()
    # print(f"Clean SRT saved at: {output_path}")
    return output_path
# Example usage




import librosa
import soundfile as sf
import subprocess

def speedup_audio_librosa(input_file, output_file, speedup_factor):
    try:
        # Load the audio file
        y, sr = librosa.load(input_file, sr=None)

        # Use time stretching to speed up audio without changing pitch
        y_stretched = librosa.effects.time_stretch(y, rate=speedup_factor)

        # Save the output with the original sample rate
        sf.write(output_file, y_stretched, sr)
        # print(f"Speed up by {speedup_factor} completed successfully: {output_file}")
    
    except Exception as e:
        gr.Warning(f"Error during speedup with Librosa: {e}")
        shutil.copy(input_file, output_file)



    
def is_ffmpeg_installed():
    if platform.system() == "Windows":
        local_ffmpeg_path = os.path.join("./ffmpeg", "ffmpeg.exe")
    else:
        local_ffmpeg_path = "ffmpeg"
    try:
        subprocess.run([local_ffmpeg_path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # print("FFmpeg is installed")
        return True,local_ffmpeg_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        # print("FFmpeg is not installed. Using 'librosa' for speedup audio in SRT dubbing")
        gr.Warning("FFmpeg is not installed. Using 'librosa' for speedup audio in SRT dubbing",duration= 20)
        return False,local_ffmpeg_path




# ffmpeg -i test.wav -filter:a "atempo=2.0" ffmpeg.wav  -y
def change_speed(input_file, output_file, speedup_factor):
    global use_ffmpeg,local_ffmpeg_path
    if use_ffmpeg:
        # print("Using FFmpeg for speedup")
        try:
            # subprocess.run([
            #         local_ffmpeg_path,
            #         "-i", input_file,
            #         "-filter:a", f"atempo={speedup_factor}",
            #         output_file,
            #         "-y"
            #     ], check=True)
            subprocess.run([
                local_ffmpeg_path,
                "-i", input_file,
                "-filter:a", f"atempo={speedup_factor}",
                output_file,
                "-y"
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            gr.Error(f"Error during speedup with FFmpeg: {e}")
            speedup_audio_librosa(input_file, output_file, speedup_factor)
    else:
        # print("Using Librosa for speedup")
        speedup_audio_librosa(input_file, output_file, speedup_factor)







class SRTDubbing:
    def __init__(self):
        pass

    @staticmethod
    def text_to_speech_srt(text, audio_path, language, actual_duration):
        tts_filename = "./cache/temp.wav"
        your_tts(text,tts_filename,actual_duration,speed=1.0)
        # Check the duration of the generated TTS audio
        tts_audio = AudioSegment.from_file(tts_filename)
        tts_duration = len(tts_audio)

        if actual_duration == 0:
            # If actual duration is zero, use the original TTS audio without modifications
            shutil.move(tts_filename, audio_path)
            return
        # If TTS audio duration is longer than actual duration, speed up the audio
        if tts_duration > actual_duration:
            speedup_factor = tts_duration / actual_duration
            speedup_filename = "./cache/speedup_temp.wav"
            change_speed(tts_filename, speedup_filename, speedup_factor)
            # Use ffmpeg to change audio speed
            # subprocess.run([
            #     "ffmpeg",
            #     "-i", tts_filename,
            #     "-filter:a", f"atempo={speedup_factor}",
            #     speedup_filename,
            #     "-y"
            # ], check=True)

            # Replace the original TTS audio with the sped-up version
            shutil.move(speedup_filename, audio_path)
        elif tts_duration < actual_duration:
            # If TTS audio duration is less than actual duration, add silence to match the duration
            silence_gap = actual_duration - tts_duration
            silence = AudioSegment.silent(duration=int(silence_gap))
            new_audio = tts_audio + silence

            # Save the new audio with added silence
            new_audio.export(audio_path, format="wav")
        else:
            # If TTS audio duration is equal to actual duration, use the original TTS audio
            shutil.move(tts_filename, audio_path)

    @staticmethod
    def make_silence(pause_time, pause_save_path):
        silence = AudioSegment.silent(duration=pause_time)
        silence.export(pause_save_path, format="wav")
        return pause_save_path

    @staticmethod
    def create_folder_for_srt(srt_file_path):
        srt_base_name = os.path.splitext(os.path.basename(srt_file_path))[0]
        random_uuid = str(uuid.uuid4())[:4]
        dummy_folder_path = f"{base_path}/dummy"
        if not os.path.exists(dummy_folder_path):
            os.makedirs(dummy_folder_path)
        folder_path = os.path.join(dummy_folder_path, f"{srt_base_name}_{random_uuid}")
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    @staticmethod
    def concatenate_audio_files(audio_paths, output_path):
        concatenated_audio = AudioSegment.silent(duration=0)
        for audio_path in audio_paths:
            audio_segment = AudioSegment.from_file(audio_path)
            concatenated_audio += audio_segment
        concatenated_audio.export(output_path, format="wav")

    def srt_to_dub(self, srt_file_path,dub_save_path,language='en'):
        result = self.read_srt_file(srt_file_path)
        new_folder_path = self.create_folder_for_srt(srt_file_path)
        join_path = []
        for i in tqdm(result):
        # for i in result:
            text = i['text']
            actual_duration = i['end_time'] - i['start_time']
            pause_time = i['pause_time']
            slient_path = f"{new_folder_path}/{i['previous_pause']}"
            self.make_silence(pause_time, slient_path)
            join_path.append(slient_path)
            tts_path = f"{new_folder_path}/{i['audio_name']}"
            self.text_to_speech_srt(text, tts_path, language, actual_duration)
            join_path.append(tts_path)
        self.concatenate_audio_files(join_path, dub_save_path)

    @staticmethod
    def convert_to_millisecond(time_str):
      if isinstance(time_str, str):
          hours, minutes, second_millisecond = time_str.split(':')
          seconds, milliseconds = second_millisecond.split(",")

          total_milliseconds = (
              int(hours) * 3600000 +
              int(minutes) * 60000 +
              int(seconds) * 1000 +
              int(milliseconds)
          )

          return total_milliseconds
    @staticmethod
    def read_srt_file(file_path):
        entries = []
        default_start = 0
        previous_end_time = default_start
        entry_number = 1
        audio_name_template = "{}.wav"
        previous_pause_template = "{}_before_pause.wav"

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # print(lines)
            for i in range(0, len(lines), 4):
                time_info = re.findall(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)', lines[i + 1])
                start_time = SRTDubbing.convert_to_millisecond(time_info[0][0])
                end_time = SRTDubbing.convert_to_millisecond(time_info[0][1])

                current_entry = {
                    'entry_number': entry_number,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': lines[i + 2].strip(),
                    'pause_time': start_time - previous_end_time if entry_number != 1 else start_time - default_start,
                    'audio_name': audio_name_template.format(entry_number),
                    'previous_pause': previous_pause_template.format(entry_number),
                }

                entries.append(current_entry)
                previous_end_time = end_time
                entry_number += 1

        with open("entries.json", "w") as file:
            json.dump(entries, file, indent=4)
        return entries
srt_voice_name="af"   
use_ffmpeg,local_ffmpeg_path = is_ffmpeg_installed()
# use_ffmpeg=False

def srt_process(srt_file_path,voice_name,custom_voicepack=None,dest_language="en"):
  global srt_voice_name,use_ffmpeg
  
  if not srt_file_path.endswith(".srt"):
      gr.Error("Please upload a valid .srt file",duration=5)
      return None
  if use_ffmpeg:
    gr.Success("Using FFmpeg for audio speedup to sync with subtitle")
  else:
    gr.Warning("Install FFmpeg to ensure high-quality audio when speeding up the audio to sync with subtitle. Default Using 'librosa' for speedup",duration= 20)

  if custom_voicepack:
    if manage_files(custom_voicepack):
        srt_voice_name = custom_voicepack
    else:
        srt_voice_name=voice_name
        gr.Warning("Upload small size .pt file only. Using the Current voice pack instead.")
  srt_dubbing = SRTDubbing()
  dub_save_path=get_subtitle_Dub_path(srt_file_path,dest_language)
  srt_dubbing.srt_to_dub(srt_file_path,dub_save_path,dest_language)
  return dub_save_path

# 
# srt_file_path="./long.srt"
# dub_audio_path=srt_process(srt_file_path)
# print(f"Audio file saved at: {dub_audio_path}")



with gr.Blocks() as demo3:

    gr.Markdown(
        """
        # Generate Audio File From Subtitle [Upload Only .srt file]
        
        To generate subtitles, you can use the [Whisper Turbo Subtitle](https://github.com/NeuralFalconYT/Whisper-Turbo-Subtitle) 
        
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/Whisper-Turbo-Subtitle/blob/main/Whisper_Turbo_Subtitle.ipynb)
        """
    )
    with gr.Row():
        with gr.Column():
            srt_file = gr.File(label='Upload .srt Subtitle File Only')
            with gr.Row():
                voice = gr.Dropdown(
                    voice_list, 
                    value='af', 
                    allow_custom_value=False, 
                    label='Voice', 
                )
            with gr.Row():
                generate_btn_ = gr.Button('Generate', variant='primary')

            with gr.Accordion('Audio Settings', open=False):
                custom_voicepack = gr.File(label='Upload Custom VoicePack .pt file')
                
            
            
        with gr.Column():
            audio = gr.Audio(interactive=False, label='Output Audio', autoplay=True)
            with gr.Accordion('Enable Autoplay', open=False):
                autoplay = gr.Checkbox(value=True, label='Autoplay')
                autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])

    # srt_file.submit(
    #     srt_process, 
    #     inputs=[srt_file, voice], 
    #     outputs=[audio]
    # )
    generate_btn_.click(
        srt_process, 
        inputs=[srt_file,voice,custom_voicepack], 
        outputs=[audio]
    )
    


display_text = "  \n".join(voice_list)

with gr.Blocks() as demo4:
    gr.Markdown(f"# Voice Names \n{display_text}")
    



import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    demo = gr.TabbedInterface([demo1, demo2,demo3,demo4], ["Batched TTS", "Multiple Speech-Type Generation","SRT Dubbing","Available Voice Names"],title="Kokoro TTS")

    demo.queue().launch(debug=debug, share=share)
    #Run on local network
    # laptop_ip="192.168.0.30"
    # port=8080
    # demo.queue().launch(debug=debug, share=share,server_name=laptop_ip,server_port=port)

if __name__ == "__main__":
    main()    


##For client side
# from gradio_client import Client
# import shutil
# import os
# os.makedirs("temp_audio", exist_ok=True)
# from gradio_client import Client
# client = Client("http://127.0.0.1:7860/")
# result = client.predict(
# 		text="Hello!!",
# 		model_name="kokoro-v0_19.pth",
# 		voice_name="af_bella",
# 		speed=1,
# 		trim=0,
# 		pad_between_segments=0,
# 		remove_silence=False,
# 		minimum_silence=0.05,
# 		api_name="/text_to_speech"
# )

# save_at=f"./temp_audio/{os.path.basename(result)}"
# shutil.move(result, save_at)
# print(f"Saved at {save_at}")
