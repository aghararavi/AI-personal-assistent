from flask import Flask, render_template, request, jsonify
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")

    system_instruction = (
            """You are a highly capable, friendly, and efficient AI Personal Assistant. 

            YOUR GOALS:
            1. Provide accurate, clear, and concise answers to any query.
            2. If a question is ambiguous, briefly ask for clarification before guessing.
            3. Be helpful, but avoid unnecessary conversational filler—get straight to the point.
            4. If asked to perform a creative task (like writing), prioritize high-quality, structured output.

            FORMATTING RULES:
            - Use Markdown for readability (bolding, lists, etc.) to make information easy to scan.
            - If the answer involves complex steps, use numbered lists.
            - If the answer is purely factual, keep it brief.
            - If you don't know the answer, admit it politely rather than hallucinating facts.

            TONE: Professional, approachable, and intelligent."""
       )
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7,
        max_output_tokens=1500
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=config
    )
    
    answer = response.text.strip()
    return jsonify({"response": answer}), 200

@app.route("/summarize", methods=["POST"])
def summarize():
    email_text = request.form.get("email")
    prompt = f"summarize the following email in 2-3 sentences: {email_text}"

    instruction = (
            """You are a highly efficient executive email assistant. Your task is to extract core information from the provided email.

            CRITICAL RULES:
            1. Do not invent, assume, or hallucinate information. Only use facts explicitly stated in the text.
            2. If the email is too short to summarize (e.g., "Thanks", "Got it"), simply output: "Short message - no summary needed."
            3. Maintain a professional, objective tone.

            Format your response STRICTLY using the following Markdown structure:

            **📌 TL;DR:**
            (One clear, direct sentence summarizing the main objective of the email)

            **🔑 Key Points:**
            - (Bullet point 1: Essential context or decisions)
            - (Bullet point 2: Supporting details)
            - (Only include a 3rd bullet if absolutely necessary)

            **✅ Action Items:**
            - (List any tasks, deadlines, or replies required by the recipient. If none, write 'None required'.)"""
        )

        
    config = types.GenerateContentConfig(
        system_instruction=instruction,
        temperature=0.3,
        max_output_tokens=1500
    )
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=config
    )
    
    summary = response.text.strip()
    return jsonify({"response": summary}), 200

if __name__ == "__main__":
    app.run(debug=True)