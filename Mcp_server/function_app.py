import os
import json
import time
import requests
import logging
import re
import tempfile
from dotenv import load_dotenv
import os, json, tempfile, time, requests
import azure.functions as func
import azure.cognitiveservices.speech as speechsdk   
from openai import AzureOpenAI
import requests
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient


load_dotenv()                                      
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY         = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = "2025-01-01-preview"
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
tool_props_1 = json.dumps([
    {
        "propertyName": "audio_url",
        "propertyType": "string",
        "description": "Public URL of a 16-kHz mono WAV file"
    }
])
 
@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="speech_to_text_diarize",
    description="Transcribe a .wav file and label each speaker turn",
    toolProperties=tool_props_1
)
def speech_to_text_diarize(context) -> str:
    payload = json.loads(context) if isinstance(context, str) else context
    audio_url = payload.get("arguments", {}).get("audio_url")
 
    # Validate URL
    if not audio_url:
        return json.dumps({"error": "Missing 'audio_url' argument"})
    if not audio_url.lower().endswith(".wav"):
        return json.dumps({"error": "Only .wav files are accepted"})
 
    # Download
    try:
        resp = requests.get(audio_url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return json.dumps({"error": f"Failed to download audio: {e}"})
 
    # Save to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp_file.write(resp.content)
        tmp_file.flush()
        wav_path = tmp_file.name
    finally:
        tmp_file.close()
 
    # Validate speech credentials
    if not speech_key or not speech_region:
        os.remove(wav_path)
        return json.dumps({"error": "SPEECH_KEY or SPEECH_REGION not set"})
 
    # Configure Speech SDK
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region
    )
    speech_config.output_format = speechsdk.OutputFormat.Detailed
   
    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config
    )
 
    segments = []
    done = False
 
    def on_transcribed(evt):
        nonlocal segments
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detail = json.loads(evt.result.json)
            speaker_id = detail.get("SpeakerId", "1")
            text = detail.get("DisplayText", evt.result.text)
            segments.append({
                "speaker": f"Speaker {speaker_id}",
                "text": text
            })
 
    def on_stop(evt):
        nonlocal done
        done = True
 
    transcriber.transcribed.connect(on_transcribed)
    transcriber.session_stopped.connect(on_stop)
    transcriber.canceled.connect(on_stop)
 
    transcriber.start_transcribing_async()
    while not done:
        time.sleep(0.5)
    transcriber.stop_transcribing_async()
 
    # Cleanup
    try:
        os.remove(wav_path)
    except OSError:
        pass
 
    if not segments:
        return json.dumps({"error": "No speech recognized"})
 
    full_text = " ".join(s["text"] for s in segments).strip()
 
 
 
    result = {
        "transcript": full_text,
        "speaker_segments": segments
    }
 
    return json.dumps(result)

tool_props_2 = json.dumps([
    {
        "propertyName": "transcript",
        "propertyType": "string",
        "description": "Full call transcript text to be analyzed"
    }
])

client = AzureOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,                                       
        azure_endpoint=AZURE_ENDPOINT,        
        api_version="2025-01-01-preview",
        api_key=AZURE_KEY,
    )
@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="analyze_full_call",
    description="Analyze a full call transcript and return structured analysis JSON",
    toolProperties=tool_props_2
)
def analyze_full_call(context: str) -> str:
    data = json.loads(context)
    args = data.get('arguments', {})
    transcript = args.get('transcript') or args.get('raw_text', "")
    
    detect_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a language-detection assistant return one word."},
            {"role": "user",   "content": f"Which language is this text in? ```{transcript[:200]}…```"}
        ],
        max_completion_tokens=10,
    )
    detected_language = detect_resp.choices[0].message.content.strip()
    
    system_prompt = (
       f"""You are an AI agent with access to tools .
       The transcript below is in {detected_language}.  
**Please produce all output (summary, scores, feedback) in {detected_language}.**
     analyze the call in that {detected_language} strictly 
     Analyze the following call transcript to evaluate agent performance in the same language as the transcript. Use the parameters defined below to provide metrics, scoring, and actionable insights:
       - Agent Name : [Name of the Agent]
       - Date : [DD/MM/YYYY]
       - Start Time : [HH:MM]
       - End Time : [HH:MM]
       - Call Type : [Inbound/Outbound]
       - Call Duration : [In Min only]

Agent Sentiment: Scoring Format: Use a scale from -1 to 1.
    - -1: Very Negative
    - -0.5: Negative
    - 0: Neutral
    - 0.5: Positive
    - 1: Very Positive

Detailed Explanation:
    - Very Negative (-1): Strong negative language, frequent complaints, frustration, or hostility.
    - Negative (-0.5): Mild negative language, some dissatisfaction, issues raised, or defensiveness.
    - Neutral (0): Neutral language, no strong emotions detected, purely informational.
    - Positive (0.5): Positive language, satisfaction, willingness to engage, or problem-solving.
    - Very Positive (1): Very positive language, high satisfaction, commendations, or empathy.

Evaluate the following transcript based on the parameters given below; scores are out of 100:
    1. Customer Satisfaction (CSAT): Perform sentiment analysis to gauge satisfaction.
    2. First Call Resolution (FCR): Identify if the issue was resolved in this call.
    3. Call Handling Time (CHT): Compare the duration with SLA of 5 minutes.
    4. Compliance Adherence:
        - Script Adherence: Did the agent follow the prescribed flow (greeting, data protection statements, closing)?
        - Compliance Adherence: Were mandatory legal/procedural steps executed (identity verification, data privacy)?
        - Resolution Quality: Was the issue resolved clearly and professionally?
    5. Engagement Level: Evaluate the agent’s tone, follow-up, and depth of resolution.

Provide a structured summary with:
    - Scoring for each metric.
    - Overall agent performance score (weighted average) always in integer.
    - Call outcome summary and actionable recommendations for improvement.

Process:
    - Analyze the transcript to extract specific answers to each query.
    - Use relevant text excerpts to support your responses.

Output:
   Respond with valid JSON only. Do not include any markdown, code fences, explanatory text, or extra fields—only the JSON object shown below, with these exact top-level fields and structure:
extract everything in the {detected_language} strictly
All the scores should always come as integers.
**Populate** "Summary" with **3–5** concise bullet points describing the main outcomes of the call.
Do not leave it empty.
Your response must be exactly one JSON object—no markdown, no code fences, no explanations, no extra keys.  
It must conform exactly to this schema strictly:
{{
  "Summary": [],
  "AgentDetails": {{
    "AgentName": "",
    "Date": "[DD/MM/YYYY]",
    "StartTime":"[HH:MM:SS]",
    "EndTime": "[HH:MM:SS]",
    "CallDuration": "[In Minutes]",
    "CallType": "",

    "CallReasonCategories": [],
    "AgentSentiment": ""
  }},
  "Entities": [],
  "keyPhrases": [],
  "Agent_Evaluation": {{
    "Total_Score": 0,
    "CustomerSatisfaction": {{ "Score": 0, "Feedback": "" }},
    "FirstCallResolution": {{ "Score": 0, "Feedback": "" }},
    "CallHandlingTime": {{ "Score": 0, "Feedback": "" }},
    "ComplianceAdherence": {{
      "ScriptAdherence": {{ "Score": 0, "Feedback": "" }},
      "ComplianceAdherence": {{ "Score": 0, "Feedback": "" }},
      "ResolutionQuality": {{ "Score": 0, "Feedback": "" }}
    }},
    "EngagementLevel": {{
      "OverAllAgentPerformanceScore": 0,
      "CallOutcomeSummaryAndActionableRecommendationsForImprovement": ""
    }}
  }}
}}
Output must be valid JSON—nothing else.
"""
    )
    user_prompt = f"Transcript: {transcript}"
   
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_completion_tokens=2048,
        frequency_penalty=0,
        presence_penalty=0,
        temperature=0
     
    )

    

    # Extract and return the JSON string
    result = response.choices[0].message.content.strip()
    # Optionally, validate that `result` is valid JSON here
    try:
        parsed = json.loads(result)
        return json.dumps({"analysis": parsed})
    except json.JSONDecodeError:
            # Second attempt: pull JSON out of ```json…``` fences (if any)
            m = re.search(r"```json\s*(\{.*\})\s*```", result, re.DOTALL)
            candidate = m.group(1).strip() if m else result
            try:
                parsed = json.loads(candidate)
                return json.dumps({"analysis": parsed})
            except json.JSONDecodeError:
                # Fallback: return raw string
                return json.dumps({
                    "error": "Invalid JSON from LLM",
                    "raw": result
                })

    



# Define the tool properties (MCP metadata)

tool_props_form = json.dumps([
    {
      "propertyName": "pdf_url",
      "propertyType": "string",
      "description": "Public URL to the PDF or image containing a call transcript"
    },
    {
      "propertyName": "model_id",
      "propertyType": "string",
      "description": "Form Recognizer model ID (e.g. prebuilt-document)"
    }
])

@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="analyze_document",
    description="Transcribe a .pdf file and Extract transcript from document and evaluate call performance",
    toolProperties=tool_props_form
)
def analyze_document(context) -> str:
    args = json.loads(context).get("arguments", {})
    url = args.get("pdf_url", "").strip()
    model_id = args.get("model_id", "prebuilt-document").strip()

    key = os.getenv("FORM_RECOGNIZER_KEY")
    endpoint = os.getenv("FORM_RECOGNIZER_ENDPOINT")

    if not key or not endpoint:
        return json.dumps({"error": "Missing FORM_RECOGNIZER_KEY or ENDPOINT"})

    try:
        client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

        poller = client.begin_analyze_document_from_url(model_id, url)
        result = poller.result()


        raw_text = "\n".join([
            line.content for page in result.pages for line in page.lines
        ]) if result.pages else ""
        extracted =[
            {
       "speaker": "",
        "text": ""
      }]
        return json.dumps({
            
            "transcript": raw_text,
            "speaker_segments": extracted,
        })

    except Exception as e:
        return json.dumps({"error": str(e)})

tool_props_converter = json.dumps([
    {
        "propertyName": "raw_text",
        "propertyType": "string",
        "description": "Arbitrary transcript string to wrap into JSON"
    }
])

@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="json_converter",
    description="Wrap a raw_text string into { transcript: ... }",
    toolProperties=tool_props_converter
)
def json_converter(context) -> str:
    """
    Expects payload:
      {
        "arguments": {
          "raw_text": "some transcript..."
        }
      }
    Returns:
      { "transcript": "some transcript..." }
    """
    # parse the JSON-RPC style context
    payload = json.loads(context)
    args = payload.get("arguments", {})

    raw = args.get("raw_text")
    if raw is None:
        return json.dumps({"error": "Expected arguments.raw_text"})
    extracted =[
            {
       "speaker": "",
        "text": ""
      }]
    return json.dumps({
            
            "transcript": raw,
            "speaker_segments": extracted,
        })


    



