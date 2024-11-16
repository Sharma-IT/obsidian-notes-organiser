import os
import openai
import anthropic
import google.generativeai as genai
import shutil
import re
import time
import argparse
from typing import Literal, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialise APIs
openai.api_key = OPENAI_API_KEY
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Constants
MAX_CONTENT_LENGTH = 1000  # Maximum characters to analyse
RETRY_DELAY = 2  # Seconds to wait between retries
MAX_RETRIES = 3  # Maximum number of retry attempts

LLMProvider = Literal["openai", "claude", "gemini"]

def get_existing_folders(directory: str) -> List[str]:
    """Get list of existing folders in the directory."""
    return [d for d in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, d)) and not d.startswith('.')]

def read_text_files(directory: str) -> List[str]:
    """Read markdown and text files from a directory."""
    return [f for f in os.listdir(directory) 
            if (f.endswith('.md') or f.endswith('.txt')) 
            and os.path.isfile(os.path.join(directory, f))]

def extract_key_content(content: str) -> str:
    """Extract key content from markdown file."""
    # Initialise the important parts
    important_parts = []
    
    # Extract YAML frontmatter if present
    yaml_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if yaml_match:
        frontmatter = yaml_match.group(1)
        important_parts.append(frontmatter)
    
    # Extract headers (lines starting with #)
    headers = re.findall(r'^#+ .*$', content, re.MULTILINE)
    if headers:
        important_parts.extend(headers[:3])  # Take first 3 headers
    
    # Get first few paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    if paragraphs:
        important_parts.append(paragraphs[0])  # Add first paragraph
    
    # Combine and truncate
    extracted = '\n'.join(important_parts)
    return extracted[:MAX_CONTENT_LENGTH] if len(extracted) > MAX_CONTENT_LENGTH else extracted

def clean_folder_name(name: str) -> str:
    """Clean the folder name by removing unwanted prefixes and symbols."""
    # Remove common prefixes
    prefixes = ['Category:', 'Category Name:', 'Folder:', 'Type:']
    cleaned = name.strip()
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove markdown symbols and clean up
    symbols = ['*', '#', '`', '_', '[', ']', '(', ')', '{', '}']
    for symbol in symbols:
        cleaned = cleaned.replace(symbol, '')
    
    # Take first word if multiple words (avoiding complex categories)
    cleaned = cleaned.split()[0] if cleaned else "misc"
    
    return cleaned.strip()

def clean_filename(filename: str) -> str:
    """Clean the filename for analysis by removing extension and special characters."""
    # Remove .md or .txt extension
    name = os.path.splitext(filename)[0]
    
    # Replace common separators with spaces
    for sep in ['-', '_', '.']:
        name = name.replace(sep, ' ')
    
    # Remove any markdown symbols
    symbols = ['*', '#', '`', '[', ']', '(', ')', '{', '}']
    for symbol in symbols:
        name = name.replace(symbol, '')
    
    return name.strip()

def retry_with_backoff(func, *args, **kwargs):
    """Retry a function with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:  # Last attempt
                raise e
            wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt + 1} failed, waiting {wait_time} seconds...")
            time.sleep(wait_time)

def analyse_content_openai(filename: str, content: str, existing_folders: List[str]) -> str:
    """Analyse content using OpenAI's GPT."""
    clean_name = clean_filename(filename)
    prompt = f"""Analyse this note's filename and key content to suggest a single-word category name for organising it. 
    If the content fits any of these existing categories, use one of them instead: {', '.join(existing_folders)}
    
    Filename: {clean_name}
    Key Content: {content}
    
    Reply with just the category name, no extra text or symbols."""
    
    response = retry_with_backoff(
        openai.ChatCompletion.create,
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['choices'][0]['message']['content']

def analyse_content_claude(filename: str, content: str, existing_folders: List[str]) -> str:
    """Analyse content using Anthropic's Claude."""
    clean_name = clean_filename(filename)
    prompt = f"""Analyse this note's filename and key content to suggest a single-word category name for organising it. 
    If the content fits any of these existing categories, use one of them instead: {', '.join(existing_folders)}
    
    Filename: {clean_name}
    Key Content: {content}
    
    Reply with just the category name, no extra text or symbols."""
    
    message = retry_with_backoff(
        claude.messages.create,
        model="claude-3-haiku-20240307",  
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content

def analyse_content_gemini(filename: str, content: str, existing_folders: List[str]) -> str:
    """Analyse content using Google's Gemini."""
    clean_name = clean_filename(filename)
    prompt = f"""Analyse this note's filename and key content to suggest a single-word category name for organising it. 
    If the content fits any of these existing categories, use one of them instead: {', '.join(existing_folders)}
    
    Filename: {clean_name}
    Key Content: {content}
    
    Reply with just the category name, no extra text or symbols."""
    
    model = genai.GenerativeModel('gemini-1.5-flash')  
    response = retry_with_backoff(
        model.generate_content,
        prompt
    )
    return response.text

def analyse_content(filename: str, content: str, existing_folders: List[str], provider: LLMProvider = "openai") -> str:
    """Analyse content using the specified LLM provider."""
    providers = {
        "openai": analyse_content_openai,
        "claude": analyse_content_claude,
        "gemini": analyse_content_gemini
    }
    
    if provider not in providers:
        raise ValueError(f"Invalid provider. Choose from: {', '.join(providers.keys())}")
    
    # Extract key content before analysis
    key_content = extract_key_content(content)
    category = providers[provider](filename, key_content, existing_folders)
    return clean_folder_name(category)

def organise_files(directory: str, provider: LLMProvider = "openai"):
    """Organise files into folders based on LLM analysis."""
    print("\nScanning existing folders...")
    existing_folders = get_existing_folders(directory)
    if existing_folders:
        print(f"Found existing folders: {', '.join(existing_folders)}\n")
    else:
        print("No existing folders found.\n")

    files = read_text_files(directory)
    if not files:
        print("No markdown or text files found to organise.")
        return

    print(f"Found {len(files)} files to organise.")
    for file in files:
        try:
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                content = f.read()
            
            folder_name = analyse_content(file, content, existing_folders, provider)
            folder_path = os.path.join(directory, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created new folder: {folder_name}")
            
            source = os.path.join(directory, file)
            destination = os.path.join(folder_path, file)
            
            # Check if we're actually moving to a different location
            if os.path.dirname(source) != os.path.dirname(destination):
                shutil.move(source, destination)
                print(f"Moved '{file}' to '{folder_name}/'")
            else:
                print(f"'{file}' is already in the correct folder")
            
            # Add new folder to existing folders list
            if folder_name not in existing_folders:
                existing_folders.append(folder_name)
                
        except Exception as e:
            print(f"Error processing '{file}': {str(e)}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Organise Obsidian notes into folders using AI-powered content analysis.'
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        help='Path to your Obsidian notes directory'
    )
    parser.add_argument(
        '-p', '--provider',
        type=str,
        choices=['openai', 'claude', 'gemini'],
        default='openai',
        help='LLM provider to use (default: openai)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually moving files'
    )
    
    args = parser.parse_args()
    
    # Get directory from argument or prompt
    notes_directory = args.directory
    if not notes_directory:
        notes_directory = input('Enter the path to your Obsidian notes directory: ')
    
    # Validate directory
    if not os.path.isdir(notes_directory):
        print(f"Error: '{notes_directory}' is not a valid directory")
        exit(1)
    
    # Convert to absolute path
    notes_directory = os.path.abspath(notes_directory)
    
    print(f"\nUsing directory: {notes_directory}")
    print(f"Using LLM provider: {args.provider}")
    if args.dry_run:
        print("Dry run mode: No files will be moved\n")
    
    try:
        organise_files(notes_directory, args.provider)
        print('\nNotes organisation completed!')
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)
