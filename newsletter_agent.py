import streamlit as st
import json
import os
import email
import pandas as pd
import re
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import html2text
import base64
import time
import sys
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.warning(f"NLTK download issue: {e}")

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
EMAIL_ACCOUNTS = {
    'Praveen Shah': 'praveenshah@gmail.com',
    'Robbie Shaw': 'robbieshaw67@gmail.com'
}

NEWSLETTER_INDICATORS = [
    'List-Unsubscribe', 'List-Id', 'List-Post',
    'unsubscribe', 'newsletter', 'digest',
    'bulletin', 'update', 'notification'
]

TOPIC_CATEGORIES = {
    'Technology': ['tech', 'software', 'ai', 'machine learning', 'data', 'digital', 'programming', 'app'],
    'Business': ['business', 'finance', 'investment', 'market', 'economy', 'startup', 'company', 'stock'],
    'Health': ['health', 'medical', 'fitness', 'wellness', 'healthcare', 'medicine', 'nutrition'],
    'Science': ['science', 'research', 'study', 'discovery', 'experiment', 'innovation', 'breakthrough'],
    'Politics': ['politics', 'government', 'policy', 'election', 'legislation', 'congress', 'senate'],
    'Education': ['education', 'learning', 'university', 'school', 'course', 'training', 'academic'],
    'Entertainment': ['entertainment', 'movie', 'music', 'tv', 'sports', 'celebrity', 'game'],
    'Travel': ['travel', 'tourism', 'destination', 'vacation', 'hotel', 'trip', 'flight'],
    'Food': ['food', 'recipe', 'cooking', 'restaurant', 'cuisine', 'dining', 'chef'],
    'Lifestyle': ['lifestyle', 'fashion', 'beauty', 'home', 'culture', 'design', 'art']
}

CREDENTIALS_FILENAME = "credentials.json"
if "installed" in st.secrets:
    try:
        with open(CREDENTIALS_FILENAME, "w") as f:
            # Must dump full dict with "installed" key at top level
            json.dump({"installed": dict(st.secrets["installed"])}, f)
        creds_path = CREDENTIALS_FILENAME
    except Exception as e:
        st.error(f"Error writing credentials: {e}")
        creds_path = None
else:
    creds_path = CREDENTIALS_FILENAME  # fallback local

class NewsletterAgent:
    def __init__(self):
        self.services = {}

    def authenticate_gmail(self, email_address):
        """Authenticate Gmail with headless OAuth (Streamlit Cloud friendly)."""
        token_file = f'tokens/{email_address.split("@")[0]}.json'
        creds = None
        if os.path.exists(token_file):
            try:
                creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            except Exception as e:
                st.warning(f"Error loading token for {email_address}: {e}")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    st.success(f"Refreshed token for {email_address}")
                except Exception as e:
                    st.error(f"Token refresh failed: {e}")
                    creds = None
            if not creds:
                if not creds_path or not os.path.exists(creds_path):
                    st.error("Gmail API credentials missing in secrets.")
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                if hasattr(flow, "run_console"):
                    st.info("Copy the URL below, open it in your browser, then paste the authorization code here.")
                    creds = flow.run_console()
                    st.success(f"Authenticated {email_address}")
                else:
                    st.error("Interactive OAuth not supported in this environment. "
                             "Run the app locally to generate tokens, then upload them to this app.")
                    return None
            try:
                os.makedirs('tokens', exist_ok=True)
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                st.warning(f"Failed to save token: {e}")

        service = build('gmail', 'v1', credentials=creds)
        self.services[email_address] = service
        return service

    def is_newsletter(self, headers):
        try:
            for ind in NEWSLETTER_INDICATORS[:3]:
                if ind in headers:
                    return True
            subject = headers.get('Subject', '').lower()
            for ind in NEWSLETTER_INDICATORS[3:]:
                if ind in subject:
                    return True
            if headers.get('Precedence', '').lower() in ('bulk', 'list'):
                return True
            return False
        except Exception:
            return False

    def extract_text(self, msg):
        text = ""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    ct = part.get_content_type()
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                        if ct == "text/plain":
                            text += body + "\n"
                        elif ct == "text/html":
                            h = html2text.HTML2Text()
                            h.ignore_links = True
                            text += h.handle(body) + "\n"
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
                    if msg.get_content_type() == "text/html":
                        h = html2text.HTML2Text()
                        h.ignore_links = True
                        text = h.handle(body)
                    else:
                        text = body
        except Exception as e:
            st.warning(f"Text extraction error: {e}")
        return text.strip()

    def summarize(self, text, max_sentences=3):
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return ' '.join(sentences)
            cleaned = [re.sub(r'[^a-zA-Z0-9\s]', '', s.lower()) for s in sentences]
            word_freq = Counter(word_tokenize(' '.join(cleaned)))
            sentence_scores = {
                i: sum(word_freq.get(w, 0) for w in word_tokenize(cleaned[i]))
                for i in range(len(sentences))
            }
            top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
            top_sentences.sort()
            return ' '.join(sentences[i] for i in top_sentences)
        except Exception:
            return text[:300] + "..." if len(text) > 300 else text

    def classify(self, text):
        text_lower = text.lower()
        topic_scores = {cat: sum(text_lower.count(k) for k in keys) for cat, keys in TOPIC_CATEGORIES.items()}
        if any(score > 0 for score in topic_scores.values()):
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            return [cat for cat, score in sorted_topics[:3] if score > 0]
        else:
            return ['General']

    def process_newsletters(self, accounts, start_date, end_date):
        all_newsletters = []
        for addr in accounts:
            st.write(f"Authenticating {addr}...")
            service = self.authenticate_gmail(addr)
            if not service:
                st.warning(f"Skipping {addr} due to failed authentication.")
                continue
            query = f"after:{start_date:%Y/%m/%d} before:{end_date:%Y/%m/%d}"
            try:
                response = service.users().messages().list(userId='me', q=query, maxResults=500).execute()
                messages = response.get('messages', [])
            except Exception as e:
                st.error(f"Error fetching messages for {addr}: {e}")
                continue

            if not messages:
                st.info(f"No newsletters found for {addr}.")
                continue

            st.write(f"Found {len(messages)} messages for {addr}")
            progress_bar = st.progress(0)
            for i, msg in enumerate(messages):
                try:
                    full_msg = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
                    headers = {h['name']: h['value'] for h in full_msg['payload'].get('headers', [])}
                    if not self.is_newsletter(headers):
                        progress_bar.progress((i+1) / len(messages))
                        continue

                    raw_msg = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
                    email_msg = email.message_from_bytes(base64.urlsafe_b64decode(raw_msg['raw']))
                    content = self.extract_text(email_msg)

                    if len(content) < 100:
                        progress_bar.progress((i+1) / len(messages))
                        continue

                    summary = self.summarize(content)
                    topics = self.classify(content)

                    try:
                        email_date = email.utils.parsedate_to_datetime(headers.get('Date', ''))
                    except Exception:
                        email_date = datetime.now()

                    newsletter = {
                        'account': addr,
                        'sender': headers.get('From', 'Unknown'),
                        'subject': headers.get('Subject', 'No Subject'),
                        'date': email_date,
                        'summary': summary,
                        'topics': ', '.join(topics),
                        'content_length': len(content)
                    }
                    all_newsletters.append(newsletter)
                except Exception as e:
                    st.warning(f"Error processing message {i + 1}: {e}")
                progress_bar.progress((i + 1) / len(messages))

        return sorted(all_newsletters, key=lambda x: x['date'], reverse=True)

def main():
    st.set_page_config(page_title="Newsletter Summarizer & Tagger", layout="wide")
    st.title("📧 Newsletter Summarizer & Tagger")
    with st.sidebar:
        st.header("Settings")
        selected_accounts = []
        for name, email_addr in EMAIL_ACCOUNTS.items():
            if st.checkbox(f"{name} ({email_addr})", value=True):
                selected_accounts.append(email_addr)
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30), max_value=datetime.now())
        end_date = st.date_input("End Date", datetime.now(), max_value=datetime.now())
        if start_date > end_date:
            st.error("Start date cannot be after end date")
            return

    if st.button("Run Agent"):
        if not selected_accounts:
            st.error("Please select at least one email account")
            return
        st.info("Processing newsletters, please wait...")
        agent = NewsletterAgent()
        newsletters = agent.process_newsletters(selected_accounts, start_date, end_date)
        if newsletters:
            df = pd.DataFrame(newsletters)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            st.success(f"Found {len(newsletters)} newsletters!")
            st.dataframe(df[['date', 'sender', 'subject', 'summary', 'topics']])
        else:
            st.warning("No newsletters found for the selected criteria.")

if __name__ == "__main__":
    main()
