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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import html2text
from bs4 import BeautifulSoup
import base64
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

# Write credentials.json from Streamlit secrets if running in Streamlit Cloud
CREDENTIALS_FILENAME = "credentials.json"
if "installed" in st.secrets:
    with open(CREDENTIALS_FILENAME, "w") as f:
        json.dump({"installed": dict(st.secrets["installed"])}, f)
    creds_path = CREDENTIALS_FILENAME
else:
    creds_path = CREDENTIALS_FILENAME  # fallback for local run

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
                        # CHANGE: use run_console instead of run_local_server!
                        creds = flow.run_console()
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
        for indicator in NEWSLETTER_INDICATORS[:3]:
            if indicator in headers:
                return True
        subject = headers.get('Subject', '').lower()
        for indicator in NEWSLETTER_INDICATORS[3:]:
            if indicator in subject:
                return True
        precedence = headers.get('Precedence', '').lower()
        if precedence in ['bulk', 'list']:
            return True
        return False

    def extract_text_content(self, msg):
        text_content = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    text_content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif part.get_content_type() == "text/html":
                    html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    h = html2text.HTML2Text()
                    h.ignore_links = True
                    text_content += h.handle(html_content)
        else:
            if msg.get_content_type() == "text/plain":
                text_content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            elif msg.get_content_type() == "text/html":
                html_content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                h = html2text.HTML2Text()
                h.ignore_links = True
                text_content = h.handle(html_content)
        return text_content.strip()

    def summarize_textrank(self, text, max_sentences=3):
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return ' '.join(sentences)
            clean_sentences = []
            for sentence in sentences:
                clean_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence.lower())
                clean_sentences.append(clean_sentence)
            words = word_tokenize(' '.join(clean_sentences))
            word_freq = Counter(words)
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                words_in_sentence = word_tokenize(sentence.lower())
                score = sum(word_freq[word] for word in words_in_sentence if word in word_freq)
                sentence_scores[i] = score
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
            top_sentences = sorted([x[0] for x in top_sentences])
            return ' '.join([sentences[i] for i in top_sentences])
        except Exception as e:
            return text[:500] + "..." if len(text) > 500 else text

    def classify_topics(self, text):
        text_lower = text.lower()
        topic_scores = {}
        for category, keywords in TOPIC_CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[category] = score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in sorted_topics[:3]] if sorted_topics else ['General']

    def process_newsletters(self, email_addresses, start_date, end_date):
        all_newsletters = []
        for email_address in email_addresses:
            st.write(f"Processing newsletters for {email_address}...")
            service = self.authenticate_gmail(email_address)
            if not service:
                continue
            try:
                query = f"after:{start_date.strftime('%Y/%m/%d')} before:{end_date.strftime('%Y/%m/%d')}"
                results = service.users().messages().list(userId='me', q=query, maxResults=500).execute()
                messages = results.get('messages', [])
                progress_bar = st.progress(0)
                for i, message in enumerate(messages):
                    try:
                        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
                        headers = {}
                        for header in msg['payload'].get('headers', []):
                            headers[header['name']] = header['value']
                        if self.is_newsletter(msg, headers):
                            raw_msg = base64.urlsafe_b64decode(
                                service.users().messages().get(
                                    userId='me', id=message['id'], format='raw'
                                ).execute()['raw']
                            )
                            email_msg = email.message_from_bytes(raw_msg)
                            content = self.extract_text_content(email_msg)
                            if content and len(content) > 100:
                                summary = self.summarize_textrank(content)
                                topics = self.classify_topics(content)
                                date_str = headers.get('Date', '')
                                try:
                                    email_date = email.utils.parsedate_to_datetime(date_str)
                                except:
                                    email_date = datetime.now()
                                newsletter_data = {
                                    'account': email_address,
                                    'sender': headers.get('From', 'Unknown'),
                                    'subject': headers.get('Subject', 'No Subject'),
                                    'date': email_date,
                                    'summary': summary,
                                    'topics': ', '.join(topics),
                                    'content_length': len(content)
                                }
                                all_newsletters.append(newsletter_data)
                        progress_bar.progress((i + 1) / len(messages))
                    except Exception as e:
                        st.write(f"Error processing message {i}: {str(e)}")
                        continue
            except Exception as e:
                st.error(f"Error processing emails for {email_address}: {str(e)}")
        all_newsletters.sort(key=lambda x: x['date'], reverse=True)
        return all_newsletters

def main():
    st.set_page_config(
        page_title="Newsletter Summarizer & Tagger",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📧 Newsletter Summarizer & Tagger Agent")
    st.markdown("**Automatically detect, summarize, and tag newsletters from your Gmail accounts**")
    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("📧 Email Accounts")
        selected_accounts = []
        for name, email in EMAIL_ACCOUNTS.items():
            if st.checkbox(f"{name} ({email})", value=True):
                selected_accounts.append(email)
        st.subheader("📅 Date Range")
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
        if start_date > end_date:
            st.error("Start date cannot be after end date")
            return
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🚀 Run Newsletter Agent", type="primary", use_container_width=True):
            if not selected_accounts:
                st.error("Please select at least one email account")
                return
            st.info("🔄 Processing newsletters... This may take a few minutes.")
            agent = NewsletterAgent()
            newsletters = agent.process_newsletters(selected_accounts, start_date, end_date)
            if newsletters:
                st.success(f"✅ Found {len(newsletters)} newsletters!")
                df = pd.DataFrame(newsletters)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False)
                st.subheader("📊 Newsletter Summary")
                col_metrics = st.columns(4)
                with col_metrics[0]:
                    st.metric("Total Newsletters", len(newsletters))
                with col_metrics[1]:
                    st.metric("Unique Senders", df['sender'].nunique())
                with col_metrics[2]:
                    st.metric("Date Range", f"{len((end_date - start_date).days)} days")
                with col_metrics[3]:
                    st.metric("Accounts Processed", len(selected_accounts))
                st.subheader("🔍 Filter Results")
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    sender_filter = st.multiselect(
                        "Filter by Sender",
                        options=df['sender'].unique(),
                        default=[]
                    )
                with filter_col2:
                    topic_filter = st.multiselect(
                        "Filter by Topic",
                        options=list(TOPIC_CATEGORIES.keys()),
                        default=[]
                    )
                filtered_df = df.copy()
                if sender_filter:
                    filtered_df = filtered_df[filtered_df['sender'].isin(sender_filter)]
                if topic_filter:
                    filtered_df = filtered_df[filtered_df['topics'].str.contains('|'.join(topic_filter), na=False)]
                st.subheader("📋 Newsletter Details")
                display_df = filtered_df.copy()
                display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
                display_df = display_df[['Date', 'sender', 'subject', 'summary', 'topics']].rename(columns={
                    'sender': 'From',
                    'subject': 'Subject',
                    'summary': 'Summary',
                    'topics': 'Topics'
                })
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Summary": st.column_config.TextColumn(width="large"),
                        "Topics": st.column_config.TextColumn(width="medium")
                    }
                )
                if st.button("📥 Download CSV", use_container_width=True):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Click to Download",
                        data=csv,
                        file_name=f"newsletters_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("❌ No newsletters found for the selected criteria.")
    with col2:
        st.subheader("📝 Instructions")
        st.markdown("""
        1. **Select Email Accounts**: Choose which accounts to process
        2. **Set Date Range**: Pick the time period to analyze
        3. **Run Agent**: Click the button to start processing
        4. **Filter Results**: Use filters to narrow down newsletters
        5. **Download Data**: Export results as CSV for further analysis

        **Note**: First time users need to authenticate with Gmail API.
        """)
        st.subheader("🔒 Privacy & Security")
        st.markdown("""
        - Uses read-only Gmail access
        - No data stored permanently
        - OAuth2 secure authentication
        - Processes data locally
        """)
if __name__ == "__main__":
    main()
