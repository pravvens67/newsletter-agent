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

# OAuth scopes and email accounts
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
EMAIL_ACCOUNTS = {
    'Praveen Shah': 'praveenshah@gmail.com',
    'Robbie Shaw': 'robbieshaw67@gmail.com'
}

# Newsletter detection indicators
NEWSLETTER_INDICATORS = [
    'List-Unsubscribe', 'List-Id', 'List-Post',
    'unsubscribe', 'newsletter', 'digest',
    'bulletin', 'update', 'notification'
]

# Topic categories for classification
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

# Write credentials.json from Streamlit secrets
CREDENTIALS_FILENAME = "credentials.json"
if "installed" in st.secrets:
    try:
        with open(CREDENTIALS_FILENAME, "w") as f:
            json.dump({"installed": dict(st.secrets["installed"])}, f)
        creds_path = CREDENTIALS_FILENAME
    except Exception as e:
        st.error(f"Error writing credentials: {e}")
        creds_path = None
else:
    creds_path = CREDENTIALS_FILENAME  # fallback for local run

class NewsletterAgent:
    def __init__(self):
        self.services = {}

    def authenticate_gmail(self, email_address):
        """Authenticate with Gmail API using headless OAuth flow"""
        try:
            token_file = f'tokens/{email_address.split("@")[0]}.json'
            creds = None
            
            # Load existing token if available
            if os.path.exists(token_file):
                try:
                    creds = Credentials.from_authorized_user_file(token_file, SCOPES)
                except Exception as e:
                    st.warning(f"Error loading existing token: {e}")
            
            # Check if credentials are valid or need refresh
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        st.success(f"Refreshed token for {email_address}")
                    except Exception as e:
                        st.error(f"Token refresh failed: {e}")
                        creds = None
                
                # If no valid credentials, start OAuth flow
                if not creds:
                    if not creds_path or not os.path.exists(creds_path):
                        st.error("Credentials file not found. Please add your Gmail API credentials to Streamlit Secrets.")
                        return None
                    
                    try:
                        st.info(f"Starting OAuth flow for {email_address}...")
                        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                        
                        # Use headless flow for Streamlit Cloud
                        creds = flow.run_local_server(port=0, open_browser=False)
                        st.success(f"Authentication successful for {email_address}")
                        
                    except Exception as e:
                        st.error(f"OAuth flow failed for {email_address}: {e}")
                        return None
                
                # Save credentials for next run
                try:
                    os.makedirs('tokens', exist_ok=True)
                    with open(token_file, 'w') as token:
                        token.write(creds.to_json())
                except Exception as e:
                    st.warning(f"Could not save token: {e}")
            
            # Build Gmail service
            service = build('gmail', 'v1', credentials=creds)
            self.services[email_address] = service
            return service
            
        except Exception as e:
            st.error(f"Authentication failed for {email_address}: {str(e)}")
            return None

    def is_newsletter(self, headers):
        """Detect if an email is a newsletter"""
        try:
            # Check headers for newsletter indicators
            for indicator in NEWSLETTER_INDICATORS[:3]:
                if indicator in headers:
                    return True
            
            # Check subject line
            subject = headers.get('Subject', '').lower()
            for indicator in NEWSLETTER_INDICATORS[3:]:
                if indicator in subject:
                    return True
            
            # Check precedence header
            precedence = headers.get('Precedence', '').lower()
            if precedence in ['bulk', 'list']:
                return True
                
            return False
        except Exception:
            return False

    def extract_text(self, msg):
        """Extract clean text content from email message"""
        text = ""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    try:
                        content_type = part.get_content_type()
                        payload = part.get_payload(decode=True)
                        
                        if payload:
                            body = payload.decode('utf-8', errors='ignore')
                            if content_type == "text/plain":
                                text += body + "\n"
                            elif content_type == "text/html":
                                try:
                                    h = html2text.HTML2Text()
                                    h.ignore_links = True
                                    h.ignore_images = True
                                    text += h.handle(body) + "\n"
                                except Exception:
                                    text += body + "\n"
                    except Exception:
                        continue
            else:
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                        content_type = msg.get_content_type()
                        
                        if content_type == "text/html":
                            try:
                                h = html2text.HTML2Text()
                                h.ignore_links = True
                                h.ignore_images = True
                                text = h.handle(body)
                            except Exception:
                                text = body
                        else:
                            text = body
                except Exception:
                    text = str(msg.get_payload())
                    
        except Exception as e:
            st.warning(f"Text extraction error: {e}")
            return ""
        
        return text.strip()

    def summarize(self, text, max_sentences=3):
        """Summarize text using simple TextRank-like algorithm"""
        try:
            if not text or len(text.strip()) < 50:
                return text[:200] + "..." if len(text) > 200 else text
            
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return ' '.join(sentences)
            
            # Clean and score sentences
            cleaned = []
            for sentence in sentences:
                cleaned_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence.lower())
                cleaned.append(cleaned_sentence)
            
            # Calculate word frequencies
            all_words = []
            for sentence in cleaned:
                all_words.extend(word_tokenize(sentence))
            
            word_freq = Counter(all_words)
            
            # Score sentences based on word frequencies
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                words_in_sentence = word_tokenize(cleaned[i])
                score = sum(word_freq.get(word, 0) for word in words_in_sentence)
                sentence_scores[i] = score
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
            top_indices = sorted([idx for idx, score in top_sentences])
            
            return ' '.join(sentences[i] for i in top_indices)
            
        except Exception as e:
            # Fallback to simple truncation
            return text[:300] + "..." if len(text) > 300 else text

    def classify_topics(self, text):
        """Classify text into topic categories"""
        try:
            if not text:
                return ['General']
            
            text_lower = text.lower()
            topic_scores = {}
            
            for category, keywords in TOPIC_CATEGORIES.items():
                score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
                if score > 0:
                    topic_scores[category] = score
            
            # Return top 3 categories or 'General' if none found
            if topic_scores:
                sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
                return [topic[0] for topic in sorted_topics[:3]]
            else:
                return ['General']
                
        except Exception:
            return ['General']

    def process_newsletters(self, accounts, start_date, end_date):
        """Process newsletters for given accounts and date range"""
        all_items = []
        
        for addr in accounts:
            try:
                st.write(f"🔐 Authenticating {addr}...")
                service = self.authenticate_gmail(addr)
                
                if not service:
                    st.error(f"Failed to authenticate {addr}")
                    continue
                
                # Build date query
                query = f"after:{start_date.strftime('%Y/%m/%d')} before:{end_date.strftime('%Y/%m/%d')}"
                st.write(f"📧 Searching emails for {addr}...")
                
                # Get message list with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = service.users().messages().list(
                            userId='me', 
                            q=query, 
                            maxResults=500
                        ).execute()
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            st.error(f"Failed to fetch messages for {addr}: {e}")
                            continue
                        time.sleep(1)
                
                messages = response.get('messages', [])
                
                if not messages:
                    st.info(f"📭 No messages found for {addr}")
                    continue
                
                st.write(f"📨 Found {len(messages)} messages for {addr}")
                progress_bar = st.progress(0)
                newsletter_count = 0
                
                for i, message in enumerate(messages):
                    try:
                        # Get message details
                        full_message = service.users().messages().get(
                            userId='me', 
                            id=message['id'], 
                            format='full'
                        ).execute()
                        
                        # Extract headers
                        headers = {}
                        for header in full_message['payload'].get('headers', []):
                            headers[header['name']] = header['value']
                        
                        # Check if it's a newsletter
                        if not self.is_newsletter(headers):
                            progress_bar.progress((i + 1) / len(messages))
                            continue
                        
                        # Get raw message for text extraction
                        raw_response = service.users().messages().get(
                            userId='me', 
                            id=message['id'], 
                            format='raw'
                        ).execute()
                        
                        raw_message = base64.urlsafe_b64decode(raw_response['raw'])
                        email_message = email.message_from_bytes(raw_message)
                        
                        # Extract text content
                        text_content = self.extract_text(email_message)
                        
                        if len(text_content.strip()) < 100:
                            progress_bar.progress((i + 1) / len(messages))
                            continue
                        
                        # Generate summary and classify topics
                        summary = self.summarize(text_content)
                        topics = self.classify_topics(text_content)
                        
                        # Parse date
                        try:
                            email_date = email.utils.parsedate_to_datetime(headers.get('Date', ''))
                        except Exception:
                            email_date = datetime.now()
                        
                        # Create newsletter data
                        newsletter_data = {
                            'account': addr,
                            'sender': headers.get('From', 'Unknown'),
                            'subject': headers.get('Subject', 'No Subject'),
                            'date': email_date,
                            'summary': summary,
                            'topics': ', '.join(topics),
                            'content_length': len(text_content)
                        }
                        
                        all_items.append(newsletter_data)
                        newsletter_count += 1
                        
                    except Exception as e:
                        st.warning(f"Error processing message {i+1}: {str(e)}")
                        
                    # Update progress
                    progress_bar.progress((i + 1) / len(messages))
                    
                    # Add small delay to avoid rate limits
                    if i % 10 == 0:
                        time.sleep(0.1)
                
                st.success(f"✅ Found {newsletter_count} newsletters for {addr}")
                
            except Exception as e:
                st.error(f"Error processing {addr}: {str(e)}")
                continue
        
        # Sort by date (newest first)
        return sorted(all_items, key=lambda x: x['date'], reverse=True)

def main():
    st.set_page_config(
        page_title="Newsletter Summarizer & Tagger",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📧 Newsletter Summarizer & Tagger")
    st.markdown("**Automatically detect, summarize, and tag newsletters from your Gmail accounts**")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Account selection
        st.subheader("📧 Email Accounts")
        selected_accounts = []
        for name, email_addr in EMAIL_ACCOUNTS.items():
            if st.checkbox(f"{name} ({email_addr})", value=True):
                selected_accounts.append(email_addr)
        
        # Date range selection
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
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Run Newsletter Agent", type="primary", use_container_width=True):
            if not selected_accounts:
                st.error("Please select at least one email account")
                return
            
            st.info("🔄 Processing newsletters... This may take a few minutes.")
            
            # Initialize and run the agent
            agent = NewsletterAgent()
            results = agent.process_newsletters(selected_accounts, start_date, end_date)
            
            if results:
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False)
                
                st.success(f"✅ Found {len(df)} newsletters!")
                
                # Display summary metrics
                st.subheader("📊 Newsletter Summary")
                col_metrics = st.columns(4)
                with col_metrics[0]:
                    st.metric("Total Newsletters", len(df))
                with col_metrics[1]:
                    st.metric("Unique Senders", df['sender'].nunique())
                with col_metrics[2]:
                    st.metric("Date Range", f"{(end_date - start_date).days} days")
                with col_metrics[3]:
                    st.metric("Accounts Processed", len(selected_accounts))
                
                # Display results table
                st.subheader("📋 Newsletter Details")
                display_df = df.copy()
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
                
                # Download functionality
                if st.button("📥 Download CSV", use_container_width=True):
                    csv = df.to_csv(index=False)
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
        4. **Review Results**: Browse newsletters with summaries and topics
        5. **Download Data**: Export results as CSV for further analysis
        
        **Note**: First-time users need to complete Gmail OAuth authentication.
        """)
        
        st.subheader("🔒 Privacy & Security")
        st.markdown("""
        - Uses read-only Gmail access
        - No data stored permanently
        - OAuth2 secure authentication
        - Processes data locally in your browser session
        """)

if __name__ == "__main__":
    main()
