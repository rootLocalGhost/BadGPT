import gradio as gr
import train, inference, data_utils, model_utils, os
from architecture import DEVICE

def update_token_count(mode):
    return f"Tokens in Dataset: {data_utils.count_tokens(mode)}"

def auto_tune(mode):
    tokens = data_utils.count_tokens(mode)
    s, b, bl = model_utils.auto_calculate_params(mode, tokens)
    return s, b, bl

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Image("assets/logo.png", width=200)
    gr.Markdown(f"# 😈 BadGPT: Omnipotent Edition\n**Backend:** {DEVICE}")
    
    with gr.Tab("📂 Data"):
        log_data = gr.Textbox(label="Processing Log")
        with gr.Row():
            btn_pdf = gr.Button("📑 Extract Text")
            btn_ocr = gr.Button("👁️ OCR Scanned PDFs")
        btn_pdf.click(data_utils.process_data, inputs=[gr.State(False)], outputs=log_data)
        btn_ocr.click(data_utils.process_data, inputs=[gr.State(True)], outputs=log_data)

    with gr.Tab("🏋️ Gym"):
        with gr.Row():
            with gr.Column():
                m_mode = gr.Radio(["scratch", "gpt2_pretrained", "gpt2_scratch"], value="scratch", label="Logic Type")
                tok_lbl = gr.Label("Tokens: 0")
                btn_tune = gr.Button("🪄 Auto-Tune My Rig")
                steps = gr.Slider(10, 50000, label="Steps", info="Total training iterations")
                batch = gr.Slider(1, 256, label="Batch Size", info="Parallel samples per step")
                block = gr.Slider(64, 2048, label="Block Size", info="Context window")
                lr = gr.Number(1e-4, label="Learning Rate")
                btn_train = gr.Button("🔥 Sweat", variant="primary")
            with gr.Column():
                train_log = gr.Textbox(label="Live Status")
                plot_loss = gr.LinePlot(x="Step", y="Loss")
                plot_vram = gr.LinePlot(x="Step", y="VRAM")
        
        btn_tune.click(auto_tune, inputs=m_mode, outputs=[steps, batch, block])
        m_mode.change(update_token_count, inputs=m_mode, outputs=tok_lbl)
        btn_train.click(train.train_generator, inputs=[m_mode, steps, gr.State(500), batch, gr.State(4), lr, block, gr.State(384), gr.State(6), gr.State(6), gr.State(0.0)], outputs=[train_log, plot_loss, gr.State(), plot_vram])

    with gr.Tab("💬 Chat"):
        status = gr.Label("Unloaded")
        btn_load = gr.Button("🧠 Load Latest Model")
        msg = gr.Textbox(label="Prompt")
        out = gr.Textbox(label="BadGPT")
        btn_load.click(inference.load_model_weights, outputs=[gr.State(), status])
        msg.submit(inference.generate_response, inputs=[gr.State(), msg, gr.State(100), gr.State(0.7)], outputs=out)

    with gr.Tab("🏰 Vault"):
        model_list = gr.Dropdown(label="Stored Models", choices=model_utils.list_models()[0] + model_utils.list_models()[1])
        btn_del = gr.Button("💥 Delete Forever", variant="stop")
        btn_del.click(model_utils.delete_model, inputs=model_list, outputs=gr.Textbox())

if __name__ == "__main__":
    app.launch()