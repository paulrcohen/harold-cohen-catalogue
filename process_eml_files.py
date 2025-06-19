#!/usr/bin/env python3
"""
Process EML files exported from Thunderbird
Extract email content, gets rid of common artifacts and saves to CSV or text files


"""

import os
import email
import csv
import re
from datetime import datetime
from email.header import decode_header
import pandas as pd
from pathlib import Path

class EMLProcessor:
    def __init__(self, eml_folder):
        self.eml_folder = Path(eml_folder)
        
    def decode_header_value(self, header_value):
        """Decode email header values that might be encoded"""
        if header_value is None:
            return ""
        
        decoded_parts = decode_header(header_value)
        decoded_string = ""
        
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                try:
                    decoded_string += part.decode(encoding or 'utf-8')
                except:
                    decoded_string += part.decode('utf-8', errors='ignore')
            else:
                decoded_string += str(part)
        
        return decoded_string.strip()
    
    def extract_body(self, msg):
        """Extract body text from email message"""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                if content_type == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode('utf-8')
                    except:
                        try:
                            body += part.get_payload(decode=True).decode('latin-1')
                        except:
                            body += str(part.get_payload())
                elif content_type == "text/html" and not body:
                    # Use HTML if no plain text found
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8')
                        # Basic HTML to text conversion
                        body = re.sub('<[^<]+?>', '', html_content)
                    except:
                        pass
        else:
            # Single part message
            try:
                body = msg.get_payload(decode=True).decode('utf-8')
            except:
                try:
                    body = msg.get_payload(decode=True).decode('latin-1')
                except:
                    body = str(msg.get_payload())
        
        # Clean up the body text
        body = re.sub(r'\n\s*\n', '\n\n', body)  # Remove excessive newlines
        body = re.sub(r' +', ' ', body)  # Remove excessive spaces
        body = re.sub(r'\r\n', '\n', body)  # Normalize line endings
        
        return body.strip()
    
    def clean_message_text(self, text):
        """Additional cleaning for message text"""
        if not text:
            return ""
        
        # Remove common email artifacts
        text = re.sub(r'^On .* wrote:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^From:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Sent:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^To:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Subject:.*$', '', text, flags=re.MULTILINE)
        
        # Remove email signatures (basic heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        signature_started = False
        
        for line in lines:
            # Common signature indicators
            if any(indicator in line.lower() for indicator in ['--', 'best regards', 'sincerely', 'sent from']):
                signature_started = True
            
            if not signature_started:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        return text.strip()
    
    def process_eml_file(self, eml_file):
        """Process a single EML file and extract content"""
        try:
            with open(eml_file, 'rb') as f:
                msg = email.message_from_bytes(f.read())
            
            # Extract headers
            subject = self.decode_header_value(msg['Subject'])
            sender = self.decode_header_value(msg['From'])
            recipient = self.decode_header_value(msg['To'])
            cc = self.decode_header_value(msg['Cc'])
            date = self.decode_header_value(msg['Date'])
            message_id = self.decode_header_value(msg['Message-ID'])
            
            # Extract body
            body = self.extract_body(msg)
            clean_body = self.clean_message_text(body)
            
            # Parse date
            try:
                parsed_date = email.utils.parsedate_to_datetime(date) if date else None
                formatted_date = parsed_date.strftime('%Y-%m-%d %H:%M:%S') if parsed_date else date
            except:
                formatted_date = date
            
            return {
                'filename': eml_file.name,
                'subject': subject,
                'sender': sender,
                'recipient': recipient,
                'cc': cc,
                'date': formatted_date,
                'date_raw': date,
                'message_id': message_id,
                'message_text': clean_body,
                'raw_body': body  # Keep original for reference
            }
            
        except Exception as e:
            print(f"❌ Error processing {eml_file}: {e}")
            return None
    
    def process_all_files(self):
        """Process all EML files in the folder"""
        eml_files = list(self.eml_folder.glob('*.eml'))
        
        if not eml_files:
            print(f"❌ No EML files found in {self.eml_folder}")
            return []
        
        print(f"📧 Found {len(eml_files)} EML files to process")
        
        emails_data = []
        for i, eml_file in enumerate(eml_files):
            print(f"📧 Processing {i+1}/{len(eml_files)}: {eml_file.name}")
            
            email_data = self.process_eml_file(eml_file)
            if email_data:
                emails_data.append(email_data)
        
        print(f"✅ Successfully processed {len(emails_data)} emails")
        return emails_data
    
    def save_to_csv(self, emails_data, output_file):
        """Save emails data to CSV file"""
        if not emails_data:
            print("❌ No email data to save")
            return
        
        df = pd.DataFrame(emails_data)
        
        # Remove the raw_body column for CSV (it's usually too long)
        if 'raw_body' in df.columns:
            df = df.drop('raw_body', axis=1)
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Saved {len(emails_data)} emails to {output_file}")
        
        # Show preview
        print(f"\n📋 Preview:")
        for i, row in df.head(3).iterrows():
            print(f"  {i+1}. {row['subject']} (from: {row['sender']})")
    
    def save_individual_text_files(self, emails_data, output_folder):
        """Save each email as individual text file"""
        if not emails_data:
            print("❌ No email data to save")
            return
        
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        for i, email_data in enumerate(emails_data):
            # Create safe filename
            subject = email_data['subject'][:50]  # Limit length
            safe_subject = re.sub(r'[^\w\s-]', '', subject)  # Remove special chars
            safe_subject = re.sub(r'[-\s]+', '-', safe_subject)  # Replace spaces with dashes
            
            filename = f"{i+1:03d}_{safe_subject}.txt"
            filepath = output_path / filename
            
            # Write email content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Subject: {email_data['subject']}\n")
                f.write(f"From: {email_data['sender']}\n")
                f.write(f"To: {email_data['recipient']}\n")
                f.write(f"Date: {email_data['date']}\n")
                f.write(f"Message-ID: {email_data['message_id']}\n")
                f.write("-" * 80 + "\n\n")
                f.write(email_data['message_text'])
        
        print(f"✅ Saved {len(emails_data)} individual text files to {output_folder}")

def main():
    """Example usage"""
    # Update this path to where Thunderbird saved your EML files
    eml_folder = "./email_related/eml_files"  # Change this to your actual folder path
    
    print("🚀 EML Processor")
    print("=" * 50)
    
    # Check if folder exists
    if not os.path.exists(eml_folder):
        print(f"❌ Folder not found: {eml_folder}")
        print("Please update the 'eml_folder' path in the script")
        return
    
    # Create processor
    processor = EMLProcessor(eml_folder)
    
    # Process all EML files
    emails_data = processor.process_all_files()
    
    if not emails_data:
        print("❌ No emails processed successfully")
        return
    
    # Save to CSV
    processor.save_to_csv(emails_data, "./email_related/machnik_emails.csv")
    
    # Save as individual text files
    processor.save_individual_text_files(emails_data, "./email_related/processed_emails")
    
    print("\n🎉 Processing complete!")
    print("Files created:")
    print("  • machnik_emails.csv - All emails in spreadsheet format")
    print("  • individual_emails/ - Each email as separate text file")

if __name__ == "__main__":
    main()

# Setup:
"""
1. Install pandas:
   pip install pandas

2. Copy your EML files to a folder (e.g., 'eml_files')

3. Update the 'eml_folder' path in main() function

4. Run the script:
   python eml_processor.py

The script will create:
- A CSV file with all email data
- Individual text files for each email
"""
