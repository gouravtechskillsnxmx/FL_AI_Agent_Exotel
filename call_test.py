"""Trigger an Exotel Calls/connect to test your flow."""
import os
from requests.auth import HTTPBasicAuth
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("EXOTEL_API_KEY")
API_TOKEN = os.environ.get("EXOTEL_API_TOKEN")
ACCOUNT_SID = os.environ.get("EXOTEL_ACCOUNT_SID") or os.environ.get("EXOTEL_SID") or os.environ.get("EXOTEL_ACCOUNT_SID") or os.environ.get("EXOTEL_SID")
SUBDOMAIN = os.environ.get("EXOTEL_SUBDOMAIN", "api.exotel.com")

url = f"https://{SUBDOMAIN}/v1/Accounts/{ACCOUNT_SID}/Calls/connect"

payload = {
    "From": "+919876543210",
    "To": "+91810xxxxxxx",
    "CallerId": os.environ.get("EXOPHONE") or "0XXXXXXXXX",
}

resp = requests.post(url, auth=HTTPBasicAuth(API_KEY, API_TOKEN), data=payload)
print(resp.status_code)
print(resp.text)
