import gradio as gr

# Replace this with your actual model inference
def predict(text):
    return f"Model output: {text}"

with gr.Blocks() as demo:
    
    # Small greeting text
    gr.Markdown("## hi there")
    
    # Big text input
    input_text = gr.Textbox(
        placeholder="share a piece of your writing",
        lines=10,
        label=""
    )
    
    # Button
    submit_btn = gr.Button("Submit")
    
    # Output (optional, but useful)
    output = gr.Textbox(label="Output")

    # Connect button → function
    submit_btn.click(fn=predict, inputs=input_text, outputs=output)

demo.launch()
