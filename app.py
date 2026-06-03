import gradio as gr


def annotate(text):
    note_id = "note_001"
    word_count = len(text.split())
    sentence_count = text.count(".") + text.count("!") + text.count("?")

    return [[note_id, word_count, sentence_count]]


with gr.Blocks() as demo:
    gr.Markdown("## Hi there. Provide a medical note below:")

    input_text = gr.Textbox(
        placeholder="Enter a medical note here...",
        lines=10,
        label="Medical Note"
    )

    submit_btn = gr.Button("Annotate Note")

    output_table = gr.Dataframe(
        headers=["Note ID", "Word Count", "Sentence Count"],
        label="Annotation Results",
        interactive=False
    )

    submit_btn.click(
        fn=annotate,
        inputs=input_text,
        outputs=output_table
    )


demo.launch()