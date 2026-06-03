import gradio as gr

# Replace this with your actual model inference
def annotate(text):
    words = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    # return a csv of note id, word count, sentence count
    note_id = "note_001"  # Replace with actual note ID
    return f"{note_id},{words},{sentences}"

with gr.Blocks() as demo:
    
    # Small greeting text
    gr.Markdown("## Hi there. Provide a medical note below:")
    
    # Big text input
    input_text = gr.Textbox(
        placeholder="Enter a medical note here...",
        lines=10,
        label=""
    )
    
    # Button
    submit_btn = gr.Button("Annotate Note")
    
    # Output (optional, but useful)
    output = gr.Textbox(label="Output")

    # display output as a table
    output_table = gr.Dataframe(headers=["Note ID", "Word Count", "Sentence Count"], label="Annotation Results")
  
    # Connect button → function
    submit_btn.click(fn=annotate, inputs=input_text, outputs=output_table)

demo.launch()
