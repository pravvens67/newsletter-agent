import streamlit as st
import json
import os
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import re
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import html2text
from bs4 import BeautifulSoup
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Download required NLTK data
nltk.download('punkt', quiet=True)

# OAuth scopes and email accounts
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
EMAIL_ACCOUNTS = {
    'Praveen Shah': 'praveenshah@gmail.com',
    'Robbie Shaw': 'robbieshaw67@gmail.com'
}

# Newsletter detection
NEWSLETTER_INDICATORS = [
    'List-Unsubscribe', 'List-Id', 'List-Post',
    'unsubscribe', 'newsletter', 'digest',
    'bulletin', 'update', 'notification'
]

# Topic categories
TOPIC_CATEGORIES = {
    'Technology': ['tech', 'software', 'ai', 'machine learning', 'data'],
    'Business': ['business', 'finance', 'investment', 'market', 'economy'],
    'Health': ['health', 'medical', 'fitness', 'wellness', 'medicine'],
    'Science': ['science', 'research', 'study', 'discovery', 'experiment'],
    'Politics': ['politics', 'government', 'policy', 'election', 'legislation'],
    'Education': ['education', 'learning', 'university', 'school', 'course'],
    'Entertainment': ['entertainment', 'movie', 'music', 'tv', 'sports'],
    'Travel': ['travel', 'tourism', 'destination', 'vacation', 'hotel'],
    'Food': ['food', 'recipe', 'cooking', 'restaurant', 'cuisine'],
    'Lifestyle': ['lifestyle', 'fashion', 'beauty', 'home', 'culture']
}

# Write credentials.json from Streamlit secrets
CREDENTIALS_FILENAME = "credentials.json"
if "installed" in st.secrets:
    with open(CREDENTIALS_FILENAME, "w") as f:
        json.dump({"installed": dict(st.secrets["installed"])}, f)
    creds_path = CREDENTIALS_FILENAME
else:
    creds_path = CREDENTIALS_FILENAME  # fallback local

class NewsletterAgent:
    def __init__(self):
        self.services = {}

    def authenticate_gmail(self, email_address):
        token_file = f'tokens/{email_address.split("@")[0]}.json'
        creds = None
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(creds_path):
                    st.error("credentials.json not found; add your Gmail API credentials to Streamlit Secrets.")
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0, open_browser=False)
            os.makedirs('tokens', exist_ok=True)
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        service = build('gmail', 'v1', credentials=creds)
        self.services[email_address] = service
        return service

    def is_newsletter(self, headers):
        for indicator in NEWSLETTER_INDICATORS[:3]:
            if indicator in headers:
                return True
        subject = headers.get('Subject', '').lower()
        for indicator in NEWSLETTER_INDICATORS[3:]:
            if indicator in subject:
                return True
        if headers.get('Precedence','').lower() in ['bulk','list']:
            return True
        return False

    def extract_text(self, msg):
        text = ""
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
                    if ct == "text/plain":
                        text += body
                    elif ct == "text/html":
                        h = html2text.HTML2Text()
                        h.ignore_links = True
                        text += h.handle(body)
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            if msg.get_content_type()=="text/html":
                h = html2text.HTML2Text(); h.ignore_links=True
                text = h.handle(body)
            else:
                text = body
        return text.strip()

    def summarize(self, text, max_sentences=3):
        sentences = sent_tokenize(text)
        if len(sentences)<=max_sentences:
            return ' '.join(sentences)
        cleaned = [re.sub(r'[^a-zA-Z0-9\s]', '', s.lower()) for s in sentences]
        freqs = Counter(word_tokenize(' '.join(cleaned)))
        scores = {i: sum(freqs[w] for w in word_tokenize(sentences[i].lower()) if w in freqs)
                  for i in range(len(sentences))}
        top = sorted(scores, key=lambda i: scores[i], reverse=True)[:max_sentences]
        return ' '.join(sentences[i] for i in sorted(top))

    def classify(self, text):
        txt = text.lower()
        scores = {cat: sum(txt.count(k) for k in keys) for cat,keys in TOPIC_CATEGORIES.items()}
        return [cat for cat,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True) if _][:3] or ['General']

    def process_newsletters(self, accounts, start_date, end_date):
        all_items=[]
        for addr in accounts:
            st.write(f"Authenticating {addr}…")
            service = self.authenticate_gmail(addr)
            if not service: continue
            q = f"after:{start_date:%Y/%m/%d} before:{end_date:%Y/%m/%d}"
            resp = service.users().messages().list(userId='me', q=q, maxResults=500).execute()
            msgs = resp.get('messages',[])
            if not msgs:
                st.info(f"No newsletters for {addr}")
                continue
            st.write(f"Found {len(msgs)} messages for {addr}")
            pb = st.progress(0)
            for i,m in enumerate(msgs):
                try:
                    full = service.users().messages().get(userId='me',id=m['id'],format='full').execute()
                    hdrs = {h['name']:h['value'] for h in full['payload'].get('headers',[])}
                    if not self.is_newsletter(hdrs): 
                        pb.progress((i+1)/len(msgs)); continue
                    raw = service.users().messages().get(userId='me',id=m['id'],format='raw').execute()['raw']
                    msg = email.message_from_bytes(base64.urlsafe_b64decode(raw))
                    txt = self.extract_text(msg)
                    if len(txt)<100:
                        pb.progress((i+1)/len(msgs)); continue
                    summary = self.summarize(txt)
                    topics = self.classify(txt)
                    try:
                        dt = email.utils.parsedate_to_datetime(hdrs.get('Date',''))
                    except:
                        dt = datetime.now()
                    all_items.append({
                        'account': addr,
                        'sender': hdrs.get('From','Unknown'),
                        'subject': hdrs.get('Subject',''),
                        'date': dt,
                        'summary': summary,
                        'topics': ', '.join(topics),
                        'length': len(txt)
                    })
                except Exception as e:
                    st.write(f"Msg {i} error: {e}")
                pb.progress((i+1)/len(msgs))
        return sorted(all_items, key=lambda x: x['date'], reverse=True)

def main():
    st.set_page_config(page_title="Newsletter Summarizer", layout="wide")
    st.title("📧 Newsletter Summarizer & Tagger")
    with st.sidebar:
        st.header("Settings")
        sel = [email for name,email in EMAIL_ACCOUNTS.items() if st.checkbox(f"{name} ({email})", True)]
        sd = st.date_input("Start Date", datetime.now()-timedelta(days=30), max_value=datetime.now())
        ed = st.date_input("End Date", datetime.now(), max_value=datetime.now())
        if sd>ed: st.error("Start after End"); return
    if st.button("Run Agent"):
        st.info("Processing…")
        agent=NewsletterAgent()
        results=agent.process_newsletters(sel, sd, ed)
        if results:
            df=pd.DataFrame(results)
            df['date']=pd.to_datetime(df['date'])
            st.success(f"Found {len(df)} newsletters")
            st.dataframe(df[['date','sender','subject','summary','topics']])
        else:
            st.warning("No newsletters found.")

if __name__=="__main__":
    main()
