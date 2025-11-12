import os
import json
import re
import ast
import asyncio
import logging
import azure.functions as func
from dotenv import load_dotenv
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from urllib.parse import urlparse
import pyodbc
from datetime import datetime, timezone

# ─── Configuration & Environment ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY      = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT     = os.getenv("DEPLOYMENT_NAME")

SSE_URL        = os.getenv("MCP_SSE_URL","http://localhost:7071/runtime/webhooks/mcp/sse")

SQL_select_unprocessed= "SELECT Transcript   FROM [xxxxxxxxxxx].[dbo].[xxxxxxxxxxxxxxx] WHERE DocID = ?" #  write query to select transcript from your own database 
# Azure Functions app
app = func.FunctionApp()

# LLM client
llm = AzureOpenAIChatCompletionClient(
    azure_deployment=DEPLOYMENT,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    model="gpt-4o",
    api_version="2025-01-01-preview",
)

# Termination condition for group chat
termination = TextMentionTermination("TERMINATE")
server_params = SseServerParams(url=SSE_URL)


# ─── Utility to clean JSON blobs ────────────────────────────────────────────────
def cleanup(js: str) -> str:
    js = js.replace("\\'", "'")
    js = re.sub(r"\\\\u([0-9A-Fa-f]{4})", r"\\u\1", js)
    return js


# ─── Helper: extract metadata via regex ─────────────────────────────────────────
def extract_metadata_from_raw_log(log_str: str):
    """
    Returns a list of dicts with keys:
      - id
      - source
      - timestamp (ISO8601)
      - type
      - models_usage: dict with prompt_tokens and completion_tokens, or None
    by regex-parsing the stringified TaskResult.
    """
    pattern = re.compile(r"""
        id='(?P<id>[^']+)'.*?
        source='(?P<source>[^']+)'.*?
        models_usage=(?P<mu>RequestUsage\(\s*prompt_tokens=(?P<prompt_tokens>\d+),\s*completion_tokens=(?P<completion_tokens>\d+)\s*\)|None).*?
        created_at=datetime\.datetime\(\s*
            (?P<year>\d+),\s*(?P<month>\d+),\s*(?P<day>\d+),\s*
            (?P<hour>\d+),\s*(?P<minute>\d+),\s*(?P<second>\d+),\s*
            (?P<microsecond>\d+),\s*tzinfo=datetime\.timezone\.\w+\)
        .*?
        type='(?P<type>[^']+)'""",
        re.DOTALL | re.VERBOSE
    )

    results = []
    for m in pattern.finditer(log_str):
        dt = datetime(
            int(m.group('year')),
            int(m.group('month')),
            int(m.group('day')),
            int(m.group('hour')),
            int(m.group('minute')),
            int(m.group('second')),
            int(m.group('microsecond')),
            tzinfo=timezone.utc
        )
        mu = m.group('mu')
        if mu and mu != 'None':
            mu_parsed = {
                "prompt_tokens":   int(m.group('prompt_tokens')),
                "completion_tokens": int(m.group('completion_tokens'))
            }
        else:
            mu_parsed = None

        results.append({
            "id":           m.group('id'),
            "source":       m.group('source'),
            "timestamp":    dt.isoformat(),
            "type":         m.group('type'),
            "models_usage": mu_parsed
        })
    return results

# ─── Core MCP pipeline ──────────────────────────────────────────────────────────
async def _run_mcp_async(url: str) -> str:
    try:
        async with McpWorkbench(server_params) as mcp_tool:
            Agent = AssistantAgent(
                    name="AssistantAgent",
                    model_client=llm,
                    workbench=mcp_tool,
                    max_tool_iterations=1,
                    handoffs=["callanalystagent"],
                    system_message=(
                        "You are AssistantAgent.\n"
                        "Your job: first find the extension of the provided URL or, if there is no URL no extension then , treat the input as a plain string.\n\n"
                        "If the extension is .wav, you become TranscriptAgent:\n"
                        "choose tool appropriately for generating transcript and speaker segments "
                        "      - 'transcript': full transcript string\n"
                        "      - 'speaker_segments': list of {speaker, text}\n"
                        "  • Do not output any extra text or markdown.\n\n"
                        "Else if the extension is .pdf, you become DocumentAgent:\n"
                        "choose analyze_document tool for analyzing document and return *only* the JSON object it produces 'transcript'.\n"
                        "          - 'transcript': full transcript string\n"
                        "          - 'speaker_segments': list of {speaker, text}\n"
                        "  • Do not output any extra text or markdown."
                        "Else (plain string input), you become JsonConvertorAgent "
                        "   call the 'json_converter' tool and return *only* a JSON object with keys :\n"
                        "      - 'transcript': full transcript string\n"
                        "      - 'speaker_segments': list of {speaker, text}\n"
                        "  • Do not output any extra text or markdown."
                    ),
)


            call_analyst = AssistantAgent(
                    name="callanalystagent",
                    model_client=llm,
                    workbench=mcp_tool,
                    max_tool_iterations=1,
                    system_message=(
                        "You are CallAnalystAgent.\n"
                        "detect the language and analyze the call in that same language "
                        "If you receive JSON with a 'transcript' key, your job is:\n"
                        "  • Call the 'analyze_full_call' tool which can analyze  the full transcript string.\n"
                        "  • return only in json format "
                        "If you receive JSON with a 'transcript' key, your job is:\n"
                        "  • Extract the 'transcript' and call the 'analyze_full_call' tool which can analyze transcript on it.\n"
                        "  • return only in json format "
                        "If you receive a plain string, your job is:\n"
                        "  • Call the 'analyze_full_call' tool with that string.\n"
                        "  • return only in json format "
                        "In all cases, return only a JSON object with a single key:\n"
                        "  'analysis' whose value is exactly the tool’s output.\n"
                        "Do not output any extra text or markdown."
          ),
)


            insights_agent = AssistantAgent(
                name="insightsagent",
                model_client=llm,
                reflect_on_tool_use=False,
                system_message=(
                    "You are insightsagent.\n"
                    "You will be passed a JSON log of the transcript and analysis results.\n"
                    "Your job is to provide a narrative explaining how the agents worked—"
                    "why specific entities/key-phrases were chosen, how sentiment was scored, "
                    "and how the summary was formed.\n"
                    "When you have finished your explanation, output the single word TERMINATE on a new line."
    ),
            )

            teams = RoundRobinGroupChat(
                [Agent, call_analyst, insights_agent],
                termination_condition=termination,
            )
            return await teams.run(task=url)

    except Exception:
        logger.exception("MCP pipeline failed")
        return json.dumps({"error": "Internal MCP error during processing"})

def run_mcp(audio_url: str) -> str:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_mcp_async(audio_url))
    finally:
        loop.close()


def extract_insights(log: str) -> str:
    tag = "source='insightsagent'"
    start_idx = log.find(tag)
    if start_idx == -1:
        raise ValueError("Couldn't find the insightsagent block in the log")

    content_idx = log.find("content=", start_idx)
    if content_idx == -1:
        raise ValueError("Couldn't find content= for insightsagent")

    quote = log[content_idx + len("content=")]
    if quote not in ("'", '"'):
        raise ValueError(f"Expected quote after content=, got {quote!r}")

    i = content_idx + len("content=") + 1
    buf = []
    while i < len(log):
        ch = log[i]
        if ch == quote and log[i - 1] != "\\":
            break
        buf.append(ch)
        i += 1

    insights = "".join(buf)
    term = "TERMINATE"
    pos = insights.find(term)
    if pos != -1:
        insights = insights[:pos]

    return insights.strip()

server = os.getenv("SQL_SERVER")
username = os.getenv("SQL_USERNAME")
password = os.getenv("SQL_PASSWORD")
database = os.getenv("SQL_DATABASE")

def databaseConn(query: str, params: tuple = None):
    """
    Execute a query and return the cursor.
    COMMITS only if it’s a modifying statement.
    Caller is responsible for cursor.close() and
    connection.close() if desired.
    """
    conn_str = (
        f"Driver={{ODBC Driver 17 for SQL Server}};"
        f"Server={server};Database={database};"
        f"UID={username};PWD={password};"
    )
    cnxn = pyodbc.connect(conn_str)
    cursor = cnxn.cursor()

    # Execute
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)

    # Only commit if it looks like an UPDATE/INSERT/DELETE
    sql = query.strip().lower()
    if sql.startswith(("insert", "update", "delete", "merge")):
        cnxn.commit()

    return cursor


@app.function_name(name="ContactCenterAnalyze")
@app.route(route="ContactCenterAnalyze", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET", "POST"])
def contact_center_analyze(req: func.HttpRequest) -> func.HttpResponse:
    # 1) Safely parse JSON body
    try:
        body = req.get_json()
    except ValueError:
        body = {}

    audio_url = req.params.get("url") or body.get("url")
    pdf_url   = req.params.get("pdf_url") or body.get("pdf_url")
    DocID    = req.params.get("docid") or body.get("docid")
    text_str=""
    
    if DocID :
        try:
            cursor = databaseConn(SQL_select_unprocessed, params=(DocID,))
            row    = cursor.fetchone()
            cursor.close()
            if row:
                text_str = row.Transcript
            else:
                return func.HttpResponse(
                    json.dumps({"error": f"No transcript found for DocID {DocID}"}),
                    status_code=404,
                    mimetype="application/json"
                )
        except Exception as e:
            logger.exception("DB fetch for text_str failed")
            return func.HttpResponse(
                json.dumps({"error": str(e)}),
                status_code=500,
                mimetype="application/json"
            )
    
    # ─── AUDIO branch ────────────────────────────────────────────────────────────
    if audio_url:
        raw_log = run_mcp(audio_url)
        
        log     = str(raw_log)
        metadata = extract_metadata_from_raw_log(log)
        
        # Extract transcript
        m1 = re.search(
            r"FunctionExecutionResult\(content='(?P<transcript>\{[\s\S]*?\})'.*?name='speech_to_text_diarize'",
            log, re.DOTALL
        )
        if not m1:
            return func.HttpResponse("Transcript JSON not found", status_code=500)

        raw_transcript = m1.group("transcript")
        try:
            transcript = json.loads(ast.literal_eval(f"'{raw_transcript}'"))
        except Exception as e:
            logger.exception("Transcript parse error")
            return func.HttpResponse(f"Transcript parse error: {e}", status_code=500)

        # Extract analysis
        m2 = re.search(
            r"FunctionExecutionResult\(content='(?P<raw>\{\"analysis\"[\s\S]*?\})'.*?name='analyze_full_call'",
            log, re.DOTALL
        )
        if not m2:
            return func.HttpResponse("Analysis JSON not found", status_code=500)

        raw_analysis = m2.group("raw")
        depth = 0
        end_idx = None
        for i, ch in enumerate(raw_analysis):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx is None:
            return func.HttpResponse("Malformed analysis JSON", status_code=500)

        analysis_payload = raw_analysis[:end_idx]
        try:
            analysis = json.loads(ast.literal_eval(f"'{analysis_payload}'"))
        except Exception as e:
            logger.exception("Analysis parse error")
            return func.HttpResponse(f"Analysis parse error: {e}", status_code=500)
        
        

        # Extract insights
        try:
            insights = extract_insights(log)
        except ValueError as e:
            return func.HttpResponse(f"Insights parse error: {e}", status_code=500)

        # Build result only once transcript, analysis, and insights exist
        result = {
            "metadata":   metadata,
            "transcript": transcript,
            "analysis": analysis,
            "insights": insights
        }
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )

    # 4) PDF branch
    if pdf_url:
        raw_log = run_mcp(pdf_url)  # now invokes analyze_document
        log     = str(raw_log)
        metadata = extract_metadata_from_raw_log(log)
        
        m3 = re.search(
            r"FunctionExecutionResult\(content='(?P<transcript>\{[\s\S]*?\})'.*?name='analyze_document'",
            log,
            re.DOTALL
        )
        if m3:
            escaped = m3.group("transcript")
            raw_text = bytes(escaped, "utf-8").decode("unicode_escape")
        try:
            transcript = json.loads(raw_text)
        except Exception as e:
            logger.exception("Transcript parse error")
            return func.HttpResponse(f"Transcript parse error: {e}", status_code=500)
                

        m = re.search(
            r"FunctionExecutionResult\(content='(?P<raw>\{\"analysis\"[\s\S]*?\})'.*?name='analyze_full_call'",
            log, re.DOTALL
        )
        if not m:
            return func.HttpResponse("Analysis JSON not found", status_code=500)

        raw_analysis = m.group("raw")
        depth = 0
        end_idx = None
        for i, ch in enumerate(raw_analysis):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx is None:
            return func.HttpResponse("Malformed analysis JSON", status_code=500)

        analysis_payload = raw_analysis[:end_idx]
        try:
            analysis = json.loads(ast.literal_eval(f"'{analysis_payload}'"))
        except Exception as e:
            logger.exception("Analysis parse error")
            return func.HttpResponse(f"Analysis parse error: {e}", status_code=500)

        # Extract insights
        try:
            insights = extract_insights(log)
        except ValueError as e:
            return func.HttpResponse(f"Insights parse error: {e}", status_code=500)
        

        # Build result only once transcript, analysis, and insights exist
        result = {
            "metadata":   metadata,
            "transcript":transcript,
            "analysis": analysis,
            "insights": insights
        }
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )


    # 5) STRING branch
    if text_str:
        raw_log = run_mcp(text_str)  # now invokes analyze_document
        log     = str(raw_log)    
        metadata = extract_metadata_from_raw_log(log)
        
        m1 = re.search(
            r"FunctionExecutionResult\(content='(?P<transcript>\{[\s\S]*?\})'.*?name='json_converter'",
            log, re.DOTALL
        )
        if not m1:
            return func.HttpResponse("Transcript JSON not found", status_code=500)

        raw_transcript = m1.group("transcript")
        try:
            transcript = json.loads(ast.literal_eval(f"'{raw_transcript}'"))
        except Exception as e:
            logger.exception("Transcript parse error")
            return func.HttpResponse(f"Transcript parse error: {e}", status_code=500)

        # Extract analysis
        m2 = re.search(
            r"FunctionExecutionResult\(content='(?P<raw>\{\"analysis\"[\s\S]*?\})'.*?name='analyze_full_call'",
            log, re.DOTALL
        )
        if not m2:
            return func.HttpResponse("Analysis JSON not found", status_code=500)

        raw_analysis = m2.group("raw")
        depth = 0
        end_idx = None
        for i, ch in enumerate(raw_analysis):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx is None:
            return func.HttpResponse("Malformed analysis JSON", status_code=500)

        analysis_payload = raw_analysis[:end_idx]
        try:
            analysis = json.loads(ast.literal_eval(f"'{analysis_payload}'"))
        except Exception as e:
            logger.exception("Analysis parse error")
            return func.HttpResponse(f"Analysis parse error: {e}", status_code=500)

        # Extract insights
        try:
            insights = extract_insights(log)
        except ValueError as e:
            return func.HttpResponse(f"Insights parse error: {e}", status_code=500)

        # Build result only once transcript, analysis, and insights exist
        result = {
            "metadata":   metadata,
            "transcript": transcript,
            "analysis": analysis,
            "insights": insights
        }
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )

        
        

    # 6) No input provided
    return func.HttpResponse(
        json.dumps({
            "error": "Please provide one of: 'url' (audio), 'pdf_url', or 'string'"
        }),
        status_code=400,
        mimetype="application/json"
    )



# ─── Audio Transcript Function ────────────────────────────────────────────────
async def handle_agentic_prompt(audio_url: str) -> dict:
    async with McpWorkbench(server_params) as mcp:
        agent = AssistantAgent(
            name="audioAgent",
            model_client=llm,
            workbench=mcp,
            reflect_on_tool_use=True,
            model_client_stream=False,
        )

        prompt = f"""
You are an AI agent with access to tools.
When given an audio URL, you must:
  1) Use the appropriate tool to get the transcript.
  2) Detect and label speaker segments—assign each utterance to “Speaker 1”, “Speaker 2”, etc., and collect them in a list.

Here is the URL: {audio_url}
"""
        result = await agent.run(task=prompt)
        raw = getattr(result, "content", None) or result

    text = str(raw)
    log = str(result)
    metadata = extract_metadata_from_raw_log(log)
    pattern = r"content='(\{(?:[^']|\\')*\})'"
    candidates = re.findall(pattern, text, flags=re.DOTALL)
    logger.info("Found %d JSON blob(s).", len(candidates))

    parsed = []
    for blob in candidates:
        fixed = cleanup(blob)
        decoded = fixed.encode('utf-8').decode('unicode_escape')
        try:
            parsed.append(json.loads(decoded))
        except json.JSONDecodeError:
            logger.warning("Failed to parse blob, skipping.")

    for j in parsed:
        if isinstance(j, dict) and "transcript" in j and "speaker_segments" in j:
            return {"metadata":metadata,
                "transcript": j}
        

    raise RuntimeError("Couldn't find the transcript blob")




@app.function_name(name="AudioTranscript")
@app.route(route="AudioTranscript", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def audio_transcript(req: func.HttpRequest) -> func.HttpResponse:
    audio_url = req.params.get("url")
    if not audio_url:
        return func.HttpResponse(
            json.dumps({"error": "Missing 'url' query parameter"}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        result = asyncio.run(handle_agentic_prompt(audio_url))
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logger.exception("AudioTranscript failed")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
