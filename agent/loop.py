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
    _search_cache: dict[str, dict] = {}
    _search_count = 0
    _eligibility_verdicts: list[dict] = []

    if trace_callback:
        trace_callback({
            "type": "pipeline_summary",
            "inferred_conditions": patient_json.get("inferred_conditions", []),
            "care_gaps": patient_json.get("care_gaps", []),
            "expanded_indications": patient_json.get("expanded_indications", []),
            "all_indications": patient_json.get("all_indications", []),
        })

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
            if _eligibility_verdicts:
                from tools.rank_with_rationale import rank_with_rationale
                if trace_callback:
                    trace_callback({"type": "tool_call", "name": "rank_with_rationale", "args": {}, "iter": iteration})
                result = rank_with_rationale(patient_json.get("patient_id", ""), _eligibility_verdicts)
                if trace_callback:
                    trace_callback({"type": "tool_result", "name": "rank_with_rationale", "result": result, "iter": iteration})
                    trace_callback({"type": "final", "content": None})
                return result
            if trace_callback:
                trace_callback({"type": "final", "content": msg.content})
            return {"outcome": "no_match", "reason": "Agent completed without finding matching trials."}

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
                if tool_call.function.name == "trial_search":
                    query = args.get("query", "")
                    if query in _search_cache:
                        result = _search_cache[query]
                    else:
                        _search_count += 1
                        if _search_count > 3:
                            result = {"trials": [], "query": query, "note": "search limit reached"}
                        else:
                            result = execute_tool(tool_call.function.name, args)
                            _search_cache[query] = result
                else:
                    result = execute_tool(tool_call.function.name, args)
                if tool_call.function.name == "check_eligibility":
                    result = _flag_resolvable(result, patient_json.get("care_gaps", []))
                    if "criteria_verdicts" in result:
                        _eligibility_verdicts.append(result)
                        if trace_callback:
                            fails = [v for v in result["criteria_verdicts"] if v.get("verdict") == "FAIL"]
                            resolvable = [v for v in fails if v.get("resolvable")]
                            trace_callback({
                                "type": "eligibility_detail",
                                "trial_id": result.get("trial_id"),
                                "overall": result.get("overall"),
                                "pass_count": result.get("pass_count", 0),
                                "total_criteria": result.get("total_criteria", 0),
                                "fails": [{"criterion": v["criterion"][:120], "rationale": v.get("rationale", ""), "resolvable": v.get("resolvable", False)} for v in fails],
                                "resolvable_count": len(resolvable),
                            })
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

    return {"outcome": "error", "reason": "Agent reached maximum iterations without producing a result."}



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
    verdict["total_criteria"] = len(verdict["criteria_verdicts"])
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