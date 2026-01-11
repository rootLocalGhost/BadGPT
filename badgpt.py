import gradio as gr
import threading
from architecture import DEVICE
import train
import inference
import os
import torch

chat_model = None
chat_status_msg = "Not Loaded"

def init_chat_model():
    global chat_model, chat_status_msg
    chat_model, chat_status_msg = inference.load_model_weights()
    return chat_status_msg

def chat_wrapper(question, max_len, temp):
    global chat_model
    if chat_model is None:
        return "⚠️ Please load a model first (Click 'Reload Weights')"
    return inference.generate_response(chat_model, question, max_len, temp)

with gr.Blocks(title="BadGPT Control Center") as app:
    gr.Markdown("# 😈 BadGPT Control Center")
    gr.Markdown(f"**Hardware:** {DEVICE.upper()} detected.")
    
    with gr.Tab("🏋️ Train (The Gym)"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Training Controls")
                steps_slider = gr.Slider(10, 10000, value=200, step=10, label="Steps to Train")
                save_slider = gr.Slider(50, 2000, value=200, step=50, label="Autosave Frequency")
                start_btn = gr.Button("🚀 Start Training", variant="primary")
            with gr.Column(scale=2):
                log_output = gr.Textbox(label="Training Log", max_lines=10, autoscroll=True)
        
        with gr.Row():
            plot_loss = gr.LinePlot(x="Step", y="Loss", title="Loss Curve")
            plot_lr = gr.LinePlot(x="Step", y="LR", title="Learning Rate")
            
        start_btn.click(
            train.train_generator, 
            inputs=[steps_slider, save_slider], 
            outputs=[log_output, plot_loss, plot_lr]
        )

    with gr.Tab("💬 Chat (Inference)"):
        with gr.Row():
            status_lbl = gr.Label(value="Status: Click Reload to check weights")
            reload_btn = gr.Button("🔄 Reload/Load Weights")
        
        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(label="User", placeholder="Type something...")
                with gr.Accordion("Parameters", open=False):
                    len_slider = gr.Slider(50, 1000, value=200, label="Max Tokens")
                    temp_slider = gr.Slider(0.1, 1.5, value=0.6, label="Temperature")
                send_btn = gr.Button("Send", variant="primary")
            with gr.Column():
                bot_output = gr.Textbox(label="BadGPT Response", lines=6)
        
        reload_btn.click(init_chat_model, outputs=status_lbl)
        send_btn.click(chat_wrapper, inputs=[user_input, len_slider, temp_slider], outputs=bot_output)

    with gr.Tab("🛠️ Utilities"):
        gr.Markdown("### Helper Tools")
        convert_btn = gr.Button("📦 Convert Checkpoint to Lite Weight")
        convert_log = gr.Textbox(label="Conversion Log")
        
        def run_conversion():
            import subprocess
            try:
                # Ensure we run in the same python environment
                result = subprocess.run([os.sys.executable, "convert_checkpoint.py"], capture_output=True, text=True)
                return result.stdout + result.stderr
            except Exception as e:
                return str(e)
                
        convert_btn.click(run_conversion, outputs=convert_log)

if __name__ == "__main__":
    app.launch(theme=gr.themes.Monochrome())