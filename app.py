from KOKORO.models import build_model
from KOKORO.utils import tts,tts_file_name
import sys
sys.path.append('.')
import torch
import gc 
print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
MODEL = build_model('./KOKORO/kokoro-v0_19.pth', device)
print("Model loaded successfully.")

def tts_maker(text,voice_name="af_bella",speed = 0.8,trim=0,pad_between=0,save_path="temp.wav",remove_silence=False,minimum_silence=50):
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

def text_to_speech(text, model_name, voice_name, speed, trim, pad_between_segments, remove_silence, minimum_silence):
    """
    Converts text to speech using the specified parameters and ensures the model is updated only if necessary.
    """
    update_status = update_model(model_name)  # Load the model only if required
    print(update_status)  # Log model loading status
    if not minimum_silence:
        minimum_silence = 0.05
    keep_silence = int(minimum_silence * 1000)
    save_at = tts_file_name(text)
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
voice_list=[]
for i in os.listdir("./KOKORO/voices"):
    voice_list.append(i.replace(".pt",""))   
# print(voice_list)     

def toggle_autoplay(autoplay):
    return gr.Audio(interactive=False, label='Output Audio', autoplay=autoplay)

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center;'>Kokoro TTS</h1>")
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
                    value='af_bella', 
                    allow_custom_value=False, 
                    label='Voice', 
                    info='Starred voices are more stable'
                )
            with gr.Row():
                generate_btn = gr.Button('Generate', variant='primary')
            with gr.Accordion('Audio Settings', open=False):
                model_name=gr.Dropdown(model_list,label="Model",value=model_list[0])
                remove_silence = gr.Checkbox(value=False, label='✂️ Remove Silence From TTS')
                minimum_silence = gr.Number(
                    label="Keep Silence Upto (In seconds)", 
                    value=0.05
                )
                speed = gr.Slider(
                    minimum=0.25, maximum=2, value=1, step=0.1, 
                    label='⚡️Speed', info='Adjust the speaking speed'
                )
                trim = gr.Slider(
                    minimum=0, maximum=1, value=0, step=0.1, 
                    label='🔪 Trim', info='How much to cut from both ends of each segment'
                )   
                pad_between = gr.Slider(
                    minimum=0, maximum=2, value=0, step=0.1, 
                    label='🔇 Pad Between', info='Silent Duration between segments [For Large Text]'
                )
                
        with gr.Column():
            audio = gr.Audio(interactive=False, label='Output Audio', autoplay=True)
            with gr.Accordion('Enable Autoplay', open=False):
                autoplay = gr.Checkbox(value=True, label='Autoplay')
                autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])

    text.submit(
        text_to_speech, 
        inputs=[text, model_name,voice, speed, trim, pad_between, remove_silence, minimum_silence], 
        outputs=[audio]
    )
    generate_btn.click(
        text_to_speech, 
        inputs=[text,model_name, voice, speed, trim, pad_between, remove_silence, minimum_silence], 
        outputs=[audio]
    )



import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
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
