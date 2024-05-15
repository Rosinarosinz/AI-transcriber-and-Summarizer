import gradio as gr
import whisper, torch   

model_size = 'medium' # ["tiny", "base", "small", "medium", "large"]
model = whisper.load_model(model_size)

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
summ_model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
summ_model.to(device)

def summarize_paragraph(paragraph):
    inputs = tokenizer.encode_plus(paragraph, return_tensors='pt', max_length=512, truncation=True, padding='longest')
    inputs = inputs.to(device)
    # Generate the summary
    summary_ids = summ_model.generate(inputs.input_ids,
                                num_beams=4,
                                max_length=150,
                                early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary

def speech_to_text(tmp_filename, audio_file):
    # print(tmp_filename)
    # print(audio_file)
    result_list = [None, None]
    summ_result_list = [None, None]
    for index, item in enumerate([tmp_filename, audio_file]):
        if item:
            result = model.transcribe(item, language="en")
            result_list[index] = result['text']

            summary = summarize_paragraph(result['text'])
            summ_result_list[index] = summary
    
    return result_list[0], summ_result_list[0], result_list[1], summ_result_list[1]


gr.Interface(
    title="Speech-to-Text Transcriber and Summarization with Deep Learning",
    fn=speech_to_text,
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="MicroPhone Recorded Audio"),
        gr.Audio(source="upload", type="filepath", label="Upload Audio File"),
        ],
        outputs=[
            gr.inputs.Textbox(label="MicroPhone Recorded Audio file, transcribe Output Text"), 
            gr.inputs.Textbox(label="MicroPhone Recorded Audio file, summary Output Text"),
            gr.inputs.Textbox(label="Upload Audio File, transcribe Output Text"),
            gr.inputs.Textbox(label="Upload Audio File, summary Output Text")
        ]
    ).launch()

