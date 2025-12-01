

"""A mushroom expert chatbot that responds to user queries about mushrooms.
"""
import gradio as gr
import random
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
import mimetypes
from sympy.physics.units import temperature

load_dotenv()
client = genai.Client()

def process_mime_file(file_path):
    """Reads a file and converts it into a Google Generative AI Part object."""
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        data = f.read()
    return types.Part.from_bytes(data=data, mime_type=mime_type)

def split_json_and_text(response_text):
    """
    Extracts the first JSON object and removes all Markdown code blocks.

    The function extracts and prints the first JSON object it finds.
    Then, it removes any and all Markdown code blocks from the
    original response text before returning the cleaned string.
    """
    # 1. Find the first JSON object and print it to the console
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            json_data = json.loads(json_str)
            print("\nExtracted JSON:", json.dumps(json_data, indent=2))
        except json.JSONDecodeError:
            print("\nWarning: Could not decode JSON from the matched string.")
    else:
        print("\nNo JSON object found in the response.")

    # 2. Clean up
    cleaned_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    final_text = re.sub(r'\{.*\}', '', cleaned_text, flags=re.DOTALL)

    return final_text.strip()

def generate_response(inputs, history):

    text = inputs['text']
    files = inputs['files']

    contents = []
    # Loop through the history and format it for the Gemini API
    for user_msg, bot_msg in history:
        history_parts = []
        if isinstance(user_msg, tuple):
            for file in user_msg:
                history_parts.append(process_mime_file(file))

        else:
            history_parts.append({"text": user_msg})

        contents.append({"role": "user", "parts": history_parts})
        if bot_msg is None:
            bot_msg = "User uploaded something other than plaintext"
        contents.append({"role": "model", "parts": [{"text": bot_msg}]})

    # Add the current user message and files to the conversation
    parts = []
    if text:
        parts.append({"text": text})

    if files:
        for file in files:
            parts.append(process_mime_file(file))


    # Add the current user message to the conversation
    contents.append({"role": "user", "parts": parts})

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                ],
                system_instruction="You are a grumpy, rugged, and well-versed old mycologist with an old man English "
                                   "accent. You know everything there is to know about mushrooms—their history, "
                                   "culinary uses, botany, and safety. You often grumble and have a no-nonsense "
                                   "attitude, but you're a font of wisdom when it comes to fungi. You're not "
                                   "knowledgeable about anything else, so if a query isn't about mushrooms, "
                                   "you'll gruffly deflect it by saying, 'Hmph. That's not a fungi-related "
                                   "inquiry. Find someone who knows about that, lad, I'm busy with my spores.' "
                                   "If a user uploads an image, provide the response in the following JSON format:"
                                   "{"
                                        "\"common_name\": \"This is the common, non-scientific name for the mushroom.\","
                                        "\"genus\": \"This is the scientific classification for a group of "
                                                            "related mushrooms.\","
                                        "\"confidence\": \"This key represents a numerical value, from 0 to 1, "
                                                            "indicating the certainty of the identification.\","
                                        "\"visible\": \"This is a list of the parts of the mushroom that were "
                                                            "clearly visible or used for identification. "
                                                            "Eg: [\"cap\", \"hymenium\", \"stipe\"]\","
                                        "\"color\": \"This is the predominant color of the mushroom\","
                                        "\"edible\": \"This is a boolean value (true or false) that indicates "
                                                        "whether the mushroom is safe to eat. true means the "
                                                        "mushroom is edible.\""
                                   "}"
                                    "And also, summarize the JSON you give"
            ),
            contents=contents
        )
    except Exception as e:
        return "No, I cannot proceed with this input."


    return split_json_and_text(response.text)

welcome_messages = [
    "Hmph. I'm busy with my research. What do you want to know about fungi, then?",
    "Aye, another one looking for mushroom knowledge. State your business.",
    "Well, don't just stand there, lad. What's your query? My time's as valuable as a rare truffle.",
    "You've found the right fellow. Ask your fungi questions, and be quick about it."
]

# Randomly select a welcome message
initial_message = random.choice(welcome_messages)


with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.ChatInterface(
        fn=generate_response,
        title="�� Your Personal Mushroom Expert ��‍��",
        multimodal=True,
        chatbot=gr.Chatbot(
            value=[
                ["Chat Started", initial_message]
            ],
            show_label=False,
        )
    )

if __name__ == "__main__":
    demo.launch(share=True)
