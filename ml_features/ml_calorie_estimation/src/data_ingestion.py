import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

edamam_api_id = os.environ["EDAMAM_API_ID"]
edamam_api_key = os.environ["EDAMAM_API_KEY"]

