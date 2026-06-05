"""Agent loop: perceive, decide, act, observe, repeat. Single agent + tools."""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from agent.prompts import AGENT_SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, execute_tool


#load environment variables
load_dotenv()

#initialize OpenAI client
client = OpenAI(
    base_url=os.getenv("NEBIUS_BASE_URL"),
    api_key=os.getenv("NEBIUS_API_KEY"),
)

#set model, default to Llama 3.3 70B Instruct unless specified in environment variables
MODEL = os.getenv("MODEL_AGENT", "meta-llama/Llama-3.3-70B-Instruct")


#define agent loop to run on one patient's data
def run_agent(patient_json: dict, trace_callback=None) -> dict:
    """Run the matching agent on one patient.

    trace_callback(event) is called for every reasoning step and tool call.
    Event types: 'agent_thinking', 'tool_call', 'tool_result', 'final'.

    trace_callback(event) returns a dictionary that contains the type of event, the content of the event to wherever run_agent is called from
    Then, the trace callback return value can be used to trace the agent's thinking and the tools it calls and the results of the tools it calls
    """

    #define the messages for the agent loop
    #The first message is the instructions + constraints for the agent, explains what the agent is allowed to do and which tools it should call when and what it is not allowed to do
    #The second message is the agent's exact tast and the provides the exact patient data (input data) for this run 
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Match this patient to relevant trials:\n{json.dumps(patient_json, indent=2)}"},
    ]

    max_iters = 25
    #iterate through the agent loop a maximum of 12 times
    for iteration in range(max_iters):
        # Call Nebius (OpenAI-compatible chat completions API) for the next LLM turn
        # Pass in model, messages, tool descriptions, and tool choice (auto means the agent will decide which tool to call)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
            parallel_tool_calls=False, #  on nebius, Llama 3.3 doesnt support multi tool call
        )
        # Get the response from the LLM
        msg = response.choices[0].message
        # Add the response to the messages list
        # the response is a dictionary that contains the role, content, and tool calls
        # if the response contains tool calls, the agent will call the tool with the arguments provided in the tool call
        # if the response does not contain tool calls, the agent will return the final answer
        # the final answer is the answer to the agent's task
        messages.append(msg.model_dump(exclude_none=True))

        # Render the agent's reasoning text (the part BETWEEN tool calls)
        if msg.content and trace_callback:
            # Update the trace callback with the agent's thinking
            trace_callback({"type": "agent_thinking", "content": msg.content, "iter": iteration})

        # No tool calls means agent is done
        # The agent is done if it does not need to call any tools to answer the task
        if not msg.tool_calls:
            # Update the trace callback with the final answer
            if trace_callback:
                trace_callback({"type": "final", "content": msg.content})
            return parse_final_output(msg.content)

        # Execute every tool call this turn (parallel if multiple)
        # The agent will call the tools with the arguments provided in the tool call
        # The tools will return the result of the tool call
        # The result of the tool call is a dictionary that contains the result of the tool call
        # The result of the tool call is a dictionary that contains the result of the tool call
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)

            if trace_callback:
                trace_callback({
                    "type": "tool_call",
                    "name": tool_call.function.name,
                    "args": args,
                    "iter": iteration,
                })

            try:
                result = execute_tool(tool_call.function.name, args)
                if tool_call.function.name == "check_eligibility":
                    result = _flag_resolvable(result, patient_json.get("care_gaps", []))
            except Exception as e:
                result = {"error": str(e)}

            if trace_callback:
                trace_callback({
                    "type": "tool_result",
                    "name": tool_call.function.name,
                    "result": result,
                    "iter": iteration,
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

            if tool_call.function.name == "rank_with_rationale" and "ranked_matches" in result:
                if trace_callback:
                    trace_callback({"type": "final", "content": None})
                return result

    return {"error": "max iterations reached", "messages": messages}


def _flag_resolvable(verdict: dict, care_gaps: list) -> dict:
    """Annotate check_eligibility results with resolvable flags by matching FAIL criteria against care gaps."""
    if "criteria_verdicts" not in verdict:
        return verdict
    gap_drugs = {g["missing_drug"].lower() for g in care_gaps}
    for v in verdict["criteria_verdicts"]:
        if v["verdict"] == "FAIL":
            criterion_lower = v["criterion"].lower()
            v["resolvable"] = any(drug in criterion_lower for drug in gap_drugs)
        else:
            v["resolvable"] = False
    verdict["resolvable_count"] = sum(1 for v in verdict["criteria_verdicts"] if v.get("resolvable"))
    return verdict


def parse_final_output(content: str) -> dict:
    """Defensively extract the JSON ranked output from the agent's final message."""
    if not content:
        return {"error": "empty final response"}
    # Try to find a JSON block, with or without markdown fence
    if "```json" in content:
        content = content.split("```json")[-1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    try:
        # Try to load the content as a JSON object and return a Python dictionary with the top 3 ranked trials
        return json.loads(content.strip())
    except json.JSONDecodeError:
        # If the content is not a valid JSON object, return the content as a string (ie model rambled)
        return {"final_text": content}