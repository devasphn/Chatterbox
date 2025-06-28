import gradio as gr
import torch
import torchaudio
import tempfile
import gc
from chatterbox.tts import ChatterboxTTS

class ChatterboxInterface:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        if self.model is None:
            print("Loading Chatterbox TTS model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        return True
    
    def generate_speech(self, text, reference_audio=None, exaggeration=0.5, 
                       cfg_weight=0.5, temperature=0.8):
        if not text.strip():
            return None, "Please enter text to synthesize."
        
        self.load_model()
        
        try:
            if reference_audio:
                wav = self.model.generate(text, audio_prompt_path=reference_audio,
                                        exaggeration=exaggeration, cfg_weight=cfg_weight,
                                        temperature=temperature)
            else:
                wav = self.model.generate(text, exaggeration=exaggeration,
                                        cfg_weight=cfg_weight, temperature=temperature)
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(temp_file.name, wav, self.model.sr)
            
            # Clear GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            return temp_file.name, "✅ Audio generated successfully!"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

# Initialize interface
tts_interface = ChatterboxInterface()

# Create Gradio UI
with gr.Blocks(title="Chatterbox TTS", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎙️ Chatterbox TTS - Voice Cloning & Speech Synthesis")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="📝 Text to Synthesize", lines=4)
            reference_audio = gr.Audio(label="🎤 Reference Audio (Optional)", 
                                     type="filepath", sources=["upload", "microphone"])
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                exaggeration = gr.Slider(0.25, 2.0, 0.5, label="🎭 Exaggeration")
                cfg_weight = gr.Slider(0.0, 1.0, 0.5, label="⚡ CFG Weight")
                temperature = gr.Slider(0.05, 2.0, 0.8, label="🌡️ Temperature")
            
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="🔊 Generated Speech")
            status_msg = gr.Textbox(label="📊 Status", interactive=False)
    
    generate_btn.click(
        tts_interface.generate_speech,
        inputs=[text_input, reference_audio, exaggeration, cfg_weight, temperature],
        outputs=[output_audio, status_msg]
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
