import json, os, threading
from agent.prompts import CHECK_ELIGIBILITY_PROMPT
from openai import OpenAI
from dotenv import load_dotenv
from data.load_fixtures import get_trial, get_patient  # add get_patient
from tools.parse_criteria import parse_criteria


#load environment variables
load_dotenv()

client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL"), api_key=os.getenv("NEBIUS_API_KEY"))
