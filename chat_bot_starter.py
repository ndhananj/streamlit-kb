ndhananj
ndhananj
Invisible

This is the start of the #aws-hackathon-genai-kb channel. 
DrFlask — 9/9/25, 10:50 PM
https://github.com/patweb99/streamlit-kb
GitHub
GitHub - patweb99/streamlit-kb: GenAI Knowledgebase Streamlit Appli...
GenAI Knowledgebase Streamlit Application. Contribute to patweb99/streamlit-kb development by creating an account on GitHub.
GitHub - patweb99/streamlit-kb: GenAI Knowledgebase Streamlit Appli...
ndhananj — 9/9/25, 10:54 PM
https://fathom.video/share/fpeLcTeyu1PJvJq6V9NfgFBGJsr2HzqR
Fathom
Online Fair Oaks AI meeting
Christopher Biddle, Robert Ehteshamzadeh, and 1 other
Chris Biddle Guitar — 9/10/25, 3:25 PM
Here is the process flow diagram I presented during last night's meeting. I think we are planning on trying the approach in page 4, right, where we could implement a bot to first intercept the user's question? If the bot can answer the request satisfactorily (i.e. the query can match up with an intent), then it processes the request. If not, it moves on to the LLM? 
Attachment file type: acrobat
Proposed_Process_Flow.pdf
51.77 KB
DrFlask — 9/10/25, 5:13 PM
https://grok.com/share/c2hhcmQtMw%3D%3D_9ff059a5-4f91-4356-a255-1adb9e5b4cf9
AWS Bedrock Python Boto3 Guide | Shared Grok Conversation
show me how to call AWS bedrock from Python using boto3 step by step
ndhananj — 9/11/25, 3:48 PM
@Chris Biddle Guitar Getting lots of data to test with is important even for RAG systems. Can you get lot of transaction example to work from?

We can always start with just a little data. But the more we have the better the overall outcomes can be.
I have some more meetings today. But I will try things out later in the evening. I already installed aws-cli on both my mac and linux machines and got access to my AWS account. 

I should be able to follow the remaining steps when I get home.
Chris Biddle Guitar — 9/11/25, 4:48 PM
@ndhananj Here is a file representing three months worth of mock checking transactions, but it only contains 68 rows (first row is column headers). All I did, though, was prompt Claude to generate these. Will it be helpful if I have it generate, say, a whole year,s worth, and upload that?
Date,Description,Amount
2024-01-01,Salary Deposit,3500.00
2024-01-02,Grocery Store - Whole Foods,-127.45
2024-01-03,Gas Station - Shell,-52.30
2024-01-04,Coffee Shop - Starbucks,-5.75
2024-01-05,Online Purchase - Amazon,-89.99
Expand
CSV_Test_Transactions.csv
3 KB
absurdist seagull — 9/11/25, 4:49 PM
what's this hackathon?
Chris Biddle Guitar — 9/11/25, 4:51 PM
@absurdist seagull https://www.meetup.com/aws-sacramento/events/310489122/?utm_medium=referral&utm_campaign=share-btn_savedevents_share_modal&utm_source=link&utm_version=v2
Meetup
First-Ever Sacramento AWS Hackathon: GenAI Knowledgebase Challenge,...
Hello Sacramento User Group!

**Something big is coming to Sacramento – and you’re invited to be part of it!**

We’re hosting **the first-ever AWS Hackathon in Sacramento!*
First-Ever Sacramento AWS Hackathon: GenAI Knowledgebase Challenge,...
ndhananj — 9/11/25, 10:23 PM
Yes. Probably.
DrFlask — Yesterday at 11:50 AM
I installed requirements into my  py312  conda environment  and it had to remove    gradio  and  pdd-cli ,   but that's fine since this will be my AWS environment.  Boto and Langchain are the most important parts.

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gradio 5.4.0 requires aiofiles<24.0,>=22.0, but you have aiofiles 24.1.0 which is incompatible.
gradio 5.4.0 requires markupsafe~=2.0, but you have markupsafe 3.0.2 which is incompatible.
datasets 3.0.1 requires fsspec[http]<=2024.6.1,>=2023.1.0, but you have fsspec 2024.10.0 which is incompatible.
pdd-cli 0.0.49 requires boto3==1.35.99, but you have boto3 1.40.29 which is incompatible.
pdd-cli 0.0.49 requires langchain_core==0.3.56, but you have langchain-core 0.3.76 which is incompatible.
pdd-cli 0.0.49 requires Requests==2.32.3, but you have requests 2.32.5 which is incompatible.
outlines 0.1.1 requires numpy<2.0.0, but you have numpy 2.1.3 which is incompatible.
Successfully installed PyPDF2-3.0.1 altair-5.5.0 blinker-1.9.0 boto3-1.40.29 botocore-1.40.29 dataclasses-json-0.6.7 httpx-sse-0.4.1 langchain-0.3.27 langchain-aws-0.2.32 langchain-community-0.3.29 langchain-core-0.3.76 langchain-text-splitters-0.3.11 narwhals-2.5.0 pydantic-settings-2.10.1 pydeck-0.9.1 requests-2.32.5 s3transfer-0.14.0 streamlit-1.49.1 watchdog-6.0.0 
ndhananj — Yesterday at 2:44 PM
I did a venv. Good success there. I will work on testing the requirements after my next round of meetings.
Chris Biddle Guitar — Yesterday at 3:58 PM
@ndhananj I had to run my app.py code in a venv as well after other options failed. Assuming that we are still planning on having an agent (i.e. a bot) intercept the user input before doing the LLM query, I am right now working on a bot using Langchain and Polars to query a CSV with the date, description, and amount fields. I was thinking, at first, building the bot using Amazon Lex, but this Langchain/Polars approach might be more straightforward and easier to integrate with the out-of-the-box streamlit-kb. What do you think?
ndhananj — Yesterday at 4:00 PM
I think that Langchain/Polars would be more natural. 

The issue might be the UX. But we have streamlit.
DrFlask — Yesterday at 4:01 PM
@Chris Biddle Guitar   what other options did you try first?
standard practice is using the javascript NPM approach  and you install a separate copy of EVERYTHING  in a BRAND NEW virtual environment  -  either venv  or conda. 
the errors I got were because I had packages that conflicted  in my existing conda environment
I like conda because each environment can be added as a Jupyter kernel ,  so I can do my development in a Jupyter notebook.
Chris Biddle Guitar — Yesterday at 4:12 PM
@ndhananj OK, I will continue with the Langchain/Polars testing and let you know where I'm at by tomorrow morning. BTW, I'm going to try to get there early tomorrow, like 9am.
DrFlask — Yesterday at 4:13 PM
do you have a number to call to be let in?
ndhananj — Yesterday at 4:14 PM
They usually post it on the door.
Chris Biddle Guitar — Yesterday at 4:14 PM
@DrFlask Let me see if I can look it up. In the past it's been one of the ladies that runs HumanBulb. Will get back to you in a bit.
DrFlask — Yesterday at 4:15 PM
thanks  🙂   my point is that if you are there early by yourself,  you could be waiting a while for someone else to let you in.
Chris Biddle Guitar — Yesterday at 4:23 PM
@DrFlask Yea, I would just get a coffee next door and wait. I just emailed Imelda Martinez (imelda@humanbulb.org) to see if she would be there as early as 9 and if she has the phone number. We will see if she responds. To answer your earlier question about what other options I had tried before the venv approach, I'd have to go back and look at my conversation with Claude. It was a bit of a whirlwind I don't recall the details. 😩
DrFlask — Yesterday at 4:33 PM
the fathom notes don't include the  claude session that you pasted in the google meet chat.
ndhananj — Yesterday at 5:18 PM
About two hours ago, the author updated it to use uv.
ndhananj — Yesterday at 5:34 PM
I need to request access to models on AWS bedrock. How long does that take?

I have access to all the open-source and Amazon ones, but not the Anthropic ones. 

I belive claude is used for the demo. 
ndhananj — Yesterday at 5:36 PM
Actually. Nevermind, I was granted access to the Anthropic ones too. It was just slower.
DrFlask — Yesterday at 5:37 PM
Image
Image
I got access to everything,  but what do I put in for model-id  on the command line?
import boto3
import json

# Initialize the Bedrock Runtime client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Define the payload (same as the CLI example)
payload = {
    "prompt": "Hello, world!",
    "max_tokens_to_sample": 50
}

try:
    # Invoke the model
    response = bedrock.invoke_model(
        modelId='anthropic.claude-v2',
        body=json.dumps(payload),
        contentType='application/json',
        accept='application/json'
    )

    # Read and parse the response
    response_body = json.loads(response['body'].read().decode('utf-8'))
    completion = response_body.get('completion', 'No completion found')
    print("Model response:", completion)

except Exception as e:
    print(f"Error: {str(e)}")
DrFlask — Yesterday at 5:45 PM
this returns: 
╰─$ python test_bedrock.py
Error: An error occurred (AccessDeniedException) when calling the InvokeModel operation: You don't have access to the model with the specified model ID.
Chris Biddle Guitar — Yesterday at 6:32 PM
@DrFlask I had gotten the same error but fixed it. There is a mismatch in the Bedrock model names. In the AWS Bedrock console, click on Model access. You will see "Titan Embeddings G1 - Text" and "Nova Micro". You need to request access to those. Once that's done, in the app.py code, change where it says "us.amazon.nova-micro-v1:0" to just "amazon.nova-micro-v1:0". Also change "us-west-2" to "us-east-1". 
ndhananj — Yesterday at 6:36 PM
It would be helpful to the other participants to make a pull request for those changes. 
Chris Biddle Guitar — Yesterday at 6:41 PM
Will do.
Chris Biddle Guitar — Yesterday at 6:52 PM
BTW, Imelda says she can let people in at 9:30, not earlier, as they need time to set up and, yes, the number will be posted on the door downstairs.
ndhananj — Yesterday at 7:11 PM
Making the changes you mentioned, I get the following warning:
/media/nithin/data2/nithin/streamlit-kb/app.py:85: LangChainDeprecationWarning: The class `BedrockEmbeddings` was deprecated in LangChain 0.2.11 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-aws package and should be used instead. To use it run `pip install -U :class:`~langchain-aws` and import as `from :class:`~langchain_aws import BedrockEmbeddings``.
  embedding_model = BedrockEmbeddings(


The chunking model worked, but the question aswering part errored in a very similar way's to @DrFlask .
ndhananj — Yesterday at 7:14 PM
My response to the cloud shell command: aws bedrock list-foundation-models --region us-east-1 returned the data attached.
{
    "modelSummaries": [
        {
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/twelvelabs.marengo-embed-2-7-v1:0",
            "modelId": "twelvelabs.marengo-embed-2-7-v1:0",
            "modelName": "Marengo Embed v2.7",... (9 KB left)
Expand
foundation_models.json
59 KB
ndhananj — Yesterday at 7:15 PM
You can see the model listed amazon.nova-micro-v1:0
Image
Chris Biddle Guitar — Yesterday at 7:33 PM
@ndhananj Here is my entire conversation with Claude on how I got the Streamlit KB working. I think I had run into a similar deprecation warning as you had shown. https://claude.ai/share/3a230dbc-7525-4272-b55b-ea871c75791c
Claude
Shared via Claude, an AI assistant from Anthropic
ndhananj — Yesterday at 8:34 PM
@Chris Biddle Guitar You may want to get Vishu to come. It seems like the bulk of the work is debugging IAM credentialling issues, and has little to do with code or LLM/RAG skills itself. 

It really is about AWS.

The same setup on openai, anthropic, or nvidia directly took like 30 seconds,  instead of hours of debug. 
Chris Biddle Guitar — Yesterday at 9:46 PM
@ndhananj OK. It's a bit late for me to reach out to him. He may show up anyway. Did you still want to participate though, even if it may be AWS-centric? Of course, we don't know for sure that whatever we come up with has to use AWS. The rules seem a bit ambiguous. I guess we will find out in the morning.
ndhananj — Yesterday at 9:48 PM
Yes. I would still participate. But I don't think we should underestimate the AWS IAM component of the work. Maybe that set up we'll be walked through when we are there. 
ndhananj — 7:21 AM
I ended up using my stuctured troubleshooting project in claude, and it made pretty quick work of the errors. I should have thought of doing that sooner. 

https://claude.ai/share/2ae741c5-8876-4947-a7a4-57e875ac12e4
Claude
Shared via Claude, an AI assistant from Anthropic
AWS Setup Learnings
10 Messages ›
ndhananj
5h ago
ndhananj — 10:56 AM
https://colab.research.google.com/drive/1KriH9SkJ2h3JVEuBDeO8XJGWJbrOR2qN?usp=sharing
Google Colab
Image
Chris Biddle Guitar — 12:15 PM
@DrFlask Here is my code as it is right now
import polars as pl
from datetime import datetime
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
Expand
message.txt
10 KB
DrFlask — 12:37 PM
import polars as pl
import duckdb
from datetime import datetime

# Load the CSV file into a Polars DataFrame
df = pl.read_csv("sample_data.csv")

# Define the filter parameters
description_filter = "Grocery"
start_date = "2025-01-01"
end_date = "2025-01-10"

# Create a DuckDB connection
con = duckdb.connect()

# Register the Polars DataFrame as a table named 'data'
con.register("data", df)

# Define the parameterized SQL query
query = """
    SELECT SUM(amount) as total_amount
    FROM data
    WHERE description ILIKE '%' || ? || '%'
    AND CAST(date AS DATE) BETWEEN ? AND ?
"""

# Execute the query with parameters
result = con.execute(query, [description_filter, start_date, end_date])

# Fetch the result (fetchone() returns a tuple with the sum)
total_amount = result.fetchone()[0] or 0.0

# Print the result
print(f"Sum of amounts for transactions containing '{description_filter}' "
      f"between {start_date} and {end_date}: {total_amount:.2f}")

# Close the DuckDB connection
con.close()
above is how to query CSV file using DuckDB
Chris Biddle Guitar — 1:44 PM
import os
import sys
import json
from datetime import datetime
from typing import Optional
import polars as pl
Expand
message.txt
16 KB
https://claude.ai/share/47280411-f4aa-424f-882b-f82287234163
Claude
Shared via Claude, an AI assistant from Anthropic
﻿
import os
import sys
import json
from datetime import datetime
from typing import Optional
import polars as pl
import boto3
from dotenv import load_dotenv

class TransactionChatbot:
    def __init__(self, csv_file_path: str, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """
        Initialize the chatbot with CSV data and AWS Bedrock client.
        
        Args:
            csv_file_path: Path to the CSV file containing transaction data
            model_id: AWS Bedrock model ID to use for LLM interactions
        """
        self.csv_file_path = csv_file_path
        self.model_id = model_id
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize AWS Bedrock client with credentials from environment
        self.initialize_aws_client()
        
        # Load and validate CSV data
        self.load_data()
    
    def initialize_aws_client(self):
        """Initialize AWS Bedrock client with credentials from environment variables."""
        try:
            # Get AWS credentials from environment variables
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')  # Default to us-east-1
            aws_session_token = os.getenv('AWS_SESSION_TOKEN')  # Optional for temporary credentials
            
            # Check if required credentials are present
            if not aws_access_key or not aws_secret_key:
                print("AWS credentials not found in environment variables.")
                print("Please ensure your .env file contains:")
                print("AWS_ACCESS_KEY_ID=your_access_key")
                print("AWS_SECRET_ACCESS_KEY=your_secret_key")
                print("AWS_DEFAULT_REGION=your_region (optional, defaults to us-east-1)")
                sys.exit(1)
            
            # Create boto3 client with explicit credentials
            session_args = {
                'aws_access_key_id': aws_access_key,
                'aws_secret_access_key': aws_secret_key,
                'region_name': aws_region
            }
            
            # Add session token if present (for temporary credentials)
            if aws_session_token:
                session_args['aws_session_token'] = aws_session_token
            
            # Create AWS session
            session = boto3.Session(**session_args)
            
            # Initialize Bedrock client
            self.bedrock = session.client('bedrock-runtime')
            
            print(f"✅ Successfully connected to AWS Bedrock in region: {aws_region}")
            
        except Exception as e:
            print(f"❌ Error initializing AWS Bedrock client: {e}")
            print("\nTroubleshooting steps:")
            print("1. Check your .env file exists and contains valid AWS credentials")
            print("2. Verify your AWS IAM user has Bedrock permissions")
            print("3. Confirm the specified region supports Bedrock")
            sys.exit(1)
    
    def load_data(self):
        """Load and validate the CSV data using Polars."""
        try:
            # Load CSV with Polars
            self.df = pl.read_csv(self.csv_file_path)
            
            # Validate required columns
            required_columns = ['date', 'description', 'amount']
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Convert date column to datetime
            self.df = self.df.with_columns([
                pl.col("date").str.to_datetime().alias("date")
            ])
            
            # Sort by date
            self.df = self.df.sort("date")
            
            print(f"Successfully loaded {len(self.df)} transactions from {self.csv_file_path}")
            print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)
    
    def call_bedrock_llm(self, prompt: str) -> str:
        """
        Call AWS Bedrock LLM with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
        try:
            # Prepare the request body for Claude
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            # Call Bedrock
            response = self.bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def parse_date_input(self, date_str: str) -> Optional[datetime]:
        """
        Parse various date formats into datetime object.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            datetime object or None if parsing fails
        """
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None
    
    def get_date_from_user(self, prompt: str) -> datetime:
        """
        Get a valid date from user input with LLM assistance.
        
        Args:
            prompt: Prompt to show the user
            
        Returns:
            Valid datetime object
        """
        while True:
            user_input = input(prompt).strip()
            
            if not user_input:
                print("Please enter a date.")
                continue
            
            # Try to parse the date directly
            parsed_date = self.parse_date_input(user_input)
            if parsed_date:
                return parsed_date
            
            # If direct parsing fails, use LLM to help
            llm_prompt = f"""
            The user entered "{user_input}" as a date. Please help parse this into a standard date format.
            If it's a valid date, respond with just the date in YYYY-MM-DD format.
            If it's not a valid date, respond with "INVALID" and suggest the correct format.
            """
            
            llm_response = self.call_bedrock_llm(llm_prompt)
            
            if "INVALID" in llm_response.upper():
                print(f"Invalid date format. {llm_response}")
                continue
            
            # Try to extract date from LLM response
            for line in llm_response.split('\n'):
                line = line.strip()
                if line and '-' in line:
                    parsed_date = self.parse_date_input(line)
                    if parsed_date:
                        return parsed_date
            
            print("Could not parse the date. Please use format: YYYY-MM-DD, MM/DD/YYYY, or similar.")
    
    def filter_transactions(self, start_date: datetime, end_date: datetime) -> pl.DataFrame:
        """
        Filter transactions within the specified date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        return filtered_df
    
    def format_transaction_summary(self, filtered_df: pl.DataFrame, start_date: datetime, end_date: datetime) -> str:
        """
        Create a formatted summary of the filtered transactions.
        
        Args:
            filtered_df: Filtered transaction data
            start_date: Start date of the range
            end_date: End date of the range
            
        Returns:
            Formatted summary string
        """
        if len(filtered_df) == 0:
            return f"No transactions found between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
        
        # Calculate summary statistics
        total_amount = filtered_df['amount'].sum()
        transaction_count = len(filtered_df)
        avg_amount = filtered_df['amount'].mean()
        
        # Format the summary
        summary = f"""
=== TRANSACTION SUMMARY ===
Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Total Transactions: {transaction_count}
Total Amount: ${total_amount:.2f}
Average Amount: ${avg_amount:.2f}

=== TRANSACTIONS ===
"""
        
        # Add individual transactions
        for row in filtered_df.iter_rows(named=True):
            date_str = row['date'].strftime('%Y-%m-%d')
            summary += f"{date_str} | ${row['amount']:>8.2f} | {row['description']}\n"
        
        return summary
    
    def generate_insights(self, filtered_df: pl.DataFrame) -> str:
        """
        Use LLM to generate insights about the filtered transactions.
        
        Args:
            filtered_df: Filtered transaction data
            
        Returns:
            LLM-generated insights
        """
        if len(filtered_df) == 0:
            return "No insights available - no transactions in the selected date range."
        
        # Prepare transaction data for LLM
        transactions_text = ""
        for row in filtered_df.iter_rows(named=True):
            date_str = row['date'].strftime('%Y-%m-%d')
            transactions_text += f"{date_str}: {row['description']} - ${row['amount']:.2f}\n"
        
        # Calculate basic statistics
        total_amount = filtered_df['amount'].sum()
        avg_amount = filtered_df['amount'].mean()
        transaction_count = len(filtered_df)
        
        llm_prompt = f"""
        Analyze these {transaction_count} transactions and provide insights:

        Summary Statistics:
        - Total Amount: ${total_amount:.2f}
        - Average Amount: ${avg_amount:.2f}
        - Number of Transactions: {transaction_count}

        Transactions:
        {transactions_text}

        Please provide:
        1. Spending patterns or trends
        2. Notable transactions (largest/smallest)
        3. Categories of expenses if identifiable
        4. Any other interesting observations

        Keep the analysis concise and practical.
        """
        
        return self.call_bedrock_llm(llm_prompt)
    
    def run(self):
        """Main chatbot loop."""
        print("🏦 Transaction Query Chatbot")
        print("=" * 50)
        
        # Show data overview
        total_transactions = len(self.df)
        date_range = f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}"
        print(f"Loaded {total_transactions} transactions from {date_range}")
        
        while True:
            print("\n" + "=" * 50)
            print("What would you like to do?")
            print("1. Query transactions by date range")
            print("2. Get AI insights about transactions")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                self.query_by_date_range()
            elif choice == '2':
                self.get_ai_insights()
            elif choice == '3':
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    def query_by_date_range(self):
        """Handle date range query functionality."""
        print("\n📅 Query Transactions by Date Range")
        print("-" * 40)
        
        # Get date range from user
        start_date = self.get_date_from_user("Enter start date: ")
        end_date = self.get_date_from_user("Enter end date: ")
        
        # Validate date range
        if start_date > end_date:
            print("Error: Start date must be before or equal to end date.")
            return
        
        # Filter transactions
        filtered_df = self.filter_transactions(start_date, end_date)
        
        # Display results
        summary = self.format_transaction_summary(filtered_df, start_date, end_date)
        print(summary)
        
        # Ask if user wants AI insights
        if len(filtered_df) > 0:
            get_insights = input("\nWould you like AI insights about these transactions? (y/n): ").strip().lower()
            if get_insights == 'y':
                print("\n🤖 AI Insights:")
                print("-" * 20)
                insights = self.generate_insights(filtered_df)
                print(insights)
    
    def get_ai_insights(self):
        """Handle AI insights for all data."""
        print("\n🤖 AI Insights for All Transactions")
        print("-" * 40)
        
        insights = self.generate_insights(self.df)
        print(insights)

def create_env_template():
    """Create a template .env file with AWS credential placeholders."""
    env_template = """# AWS Credentials for Bedrock Access
# Replace the placeholder values with your actual AWS credentials

AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# Optional: For temporary credentials (like from AWS STS)
# AWS_SESSION_TOKEN=your_session_token_here

# Security Note: Never commit this file to version control!
# Add .env to your .gitignore file
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_template)
        print("✅ Created .env template file")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def main():
    """Main function to run the chatbot."""
    print("🏦 Welcome to the Transaction Query Chatbot!")
    print("=" * 50)
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Creating a template...")
        create_env_template()
        print("Please fill in your AWS credentials in the .env file and run the script again.")
        return
    
    # Get CSV file path
    csv_file = input("Enter the path to your CSV file: ").strip()
    
    if not csv_file:
        print("Error: Please provide a CSV file path.")
        return
    
    # Optional: Get model ID
    model_choice = input("Enter Bedrock model ID (or press Enter for default Claude-3 Sonnet): ").strip()
    model_id = model_choice if model_choice else "anthropic.claude-3-sonnet-20240229-v1:0"
    
    try:
        # Create and run chatbot
        chatbot = TransactionChatbot(csv_file, model_id)
        chatbot.run()
    except KeyboardInterrupt:
        print("\n\nChatbot interrupted. Goodbye! 👋")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
message.txt
16 KB