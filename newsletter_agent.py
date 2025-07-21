import streamlit as st
import json

# Write credentials.json from Streamlit secrets if running in Streamlit Cloud
if "installed" in st.secrets:
    with open("credentials.json", "w") as f:
        json.dump(dict(st.secrets["installed"]), f)
    creds_path = "credentials.json"
else:
    creds_path = "credentials.json"  # fallback for local run with physical file

import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import re
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import html2text
from bs4 import BeautifulSoup
import base64
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# For NLTK use in Streamlit Cloud
nltk.download('punkt', quiet=True)

# Configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
EMAIL_ACCOUNTS = {
    'Praveen Shah': 'praveenshah@gmail.com',
    'Robbie Shaw': 'robbieshaw67@gmail.com'
}

# Newsletter detection patterns
NEWSLETTER_INDICATORS = [
    'List-Unsubscribe', 'List-Id', 'List-Post', 'unsubscribe',
    'newsletter', 'digest', 'bulletin', 'update', 'notification'
]

# Topic categories for classification
TOPIC_CATEGORIES = {
    'Technology': ['tech', 'software', 'ai', 'machine learning', 'data', 'digital', 'programming', 'app'],
    'Business': ['business', 'finance', 'investment', 'market', 'economy', 'startup', 'company', 'stock'],
    'Health': ['health', 'medical', 'fitness', 'wellness', 'healthcare', 'medicine', 'nutrition'],
    'Science': ['science', 'research', 'study', 'discovery', 'experiment', 'innovation', 'breakthrough'],
    'Politics': ['politics', 'government', 'policy', 'election', 'legislation', 'congress', 'senate'],
    'Education': ['education', 'learning', 'university', 'school', 'course', 'training', 'academic'],
    'Entertainment': ['entertainment', 'movie', 'music', 'tv', 'celebrity', 'sports', 'game'],
    'Travel': ['travel', 'tourism', 'destination', 'vacation', 'trip', 'hotel', 'flight'],
    'Food': ['food', 'recipe', 'cooking', 'restaurant', 'chef', 'cuisine', 'dining'],
    'Lifestyle': ['lifestyle', 'fashion', 'beauty', 'home', 'design', 'culture', 'art']
}

class NewsletterAgent:
    def __init__(self):
        self.services = {}
        self.newsletters = []

    def authenticate_gmail(self, email_address):
        """Authenticate with Gmail API"""
        try:
            creds = None
            token_file = f'tokens/{email_address.split("@")[0]}.json'

            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if os.path.exists(creds_path):
                        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                        creds = flow.run_local_server(port=0)
                    else:
                        st.error("credentials.json file not found. Please upload your Gmail API credentials.")
                        return None

                # Save credentials for next run
                os.makedirs('tokens', exist_ok=True)
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())

            service = build('gmail', 'v1', credentials=creds)
            self.services[email_address] = service
            return service

        except Exception as e:
            st.error(f"Authentication failed for {email_address}: {str(e)}")
            return None

    def is_newsletter(self, message_data, headers):
        """Detect if email is a newsletter"""
        # Check headers for newsletter indicators
        for indicator in NEWSLETTER_INDICATORS
