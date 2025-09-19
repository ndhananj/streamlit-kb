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
