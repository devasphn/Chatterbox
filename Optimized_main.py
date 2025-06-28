import gradio as gr
import torch
import torchaudio
import tempfile
import gc
import os
from chatterbox.tts import ChatterboxTTS

class ChatterboxInterface:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        if self.model is None:
            print("Loading Chatterbox TTS model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            
            # Compile for 2-4x speedup
            try:
                print("Compiling model for faster inference...")
                self.model.t3.compile(mode="reduce-overhead")
                print("✅ Model compilation successful!")
            except Exception as e:
                print(f"⚠️ Compilation failed: {e}")
                
        return True
    
    def generate_speech(self, text, reference_audio=None, exaggeration=0.5, 
                       cfg_weight=0.3, temperature=0.5, max_length=150):
        if not text.strip():
            return None, "Please enter text to synthesize."
        
        # Limit text for faster generation
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        self.load_model()
        
        try:
            # Optimized generation settings
            if reference_audio:
                wav = self.model.generate(
                    text, 
                    audio_prompt_path=reference_audio,
                    exaggeration=exaggeration, 
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
            else:
                wav = self.model.generate(
                    text, 
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight, 
                    temperature=temperature
                )
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(temp_file.name, wav, self.model.sr)
            
            # Memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            return temp_file.name, "✅ Audio generated successfully!"
            
        except Exception as e:
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            return None, f"❌ Error: {str(e)}"

# Initialize interface
tts_interface = ChatterboxInterface()

# Optimized UI
with gr.Blocks(title="Chatterbox TTS - Fast Mode", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎙️ Chatterbox TTS - Speed Optimized")
    gr.Markdown("⚡ **Fast Mode**: Limited to 150 characters for quick generation")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="📝 Text (Max 150 chars)", 
                lines=2,
                placeholder="Short text for fast generation..."
            )
            reference_audio = gr.Audio(
                label="🎤 Reference Audio (Optional)", 
                type="filepath", 
                sources=["upload", "microphone"]
            )
            
            with gr.Accordion("⚙️ Speed Settings", open=True):
                exaggeration = gr.Slider(0.3, 1.0, 0.5, label="🎭 Exaggeration")
                cfg_weight = gr.Slider(0.1, 0.5, 0.3, label="⚡ CFG Weight (Lower = Faster)")
                temperature = gr.Slider(0.3, 1.0, 0.5, label="🌡️ Temperature")
                max_length = gr.Slider(50, 200, 150, label="📏 Max Text Length")
            
            generate_btn = gr.Button("🚀 Generate (Fast)", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="🔊 Generated Speech")
            status_msg = gr.Textbox(label="📊 Status", interactive=False)
    
    generate_btn.click(
        tts_interface.generate_speech,
        inputs=[text_input, reference_audio, exaggeration, cfg_weight, temperature, max_length],
        outputs=[output_audio, status_msg]
    )

if __name__ == "__main__":
    # Memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print("🚀 Starting Fast Mode Chatterbox TTS...")
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
