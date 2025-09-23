import os
import logging
import requests
from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from rich.console import Console
from rich.markdown import Markdown
import ollama
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # fallback to "ollama"
USE_OPENAI = os.getenv("USE_OPENAI", "False").lower() in ("true", "1", "yes")

# Requests session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

# Prompts
SYSTEM_PROMPT = """You are an assistant that analyzes the contents of a website
and provides a short summary, ignoring text that might be navigation related.
Respond in markdown.
"""


# Website Class
class Website:
    def __init__(self, url: str, max_chars: int = 5000):
        self.url = url
        self.title = "No title found"
        self.text = ""

        try:
            response = session.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return

        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string.strip() if soup.title else "No title found"

        # Remove irrelevant elements
        for tag in soup(["script", "style", "img", "input", "nav", "footer", "aside", "header"]):
            tag.decompose()

        self.text = soup.get_text(separator="\n", strip=True)

        # Truncate if too long
        if len(self.text) > max_chars:
            logger.warning(f"Text too long ({len(self.text)} chars). Truncating.")
            self.text = self.text[:max_chars]

# website = Website("https://edwarddonner.com")
# print(website.url)
# print(website.title)
# print(website.text)


# Prompt Builders
def user_prompt_for(website: Website) -> str:
    return (
        f"You are looking at a website titled {website.title}\n"
        "The contents of this website are as follows. "
        "Please provide a short summary of this website in markdown. "
        "If it includes news or announcements, then summarize these too.\n\n"
        f"{website.text}"
    )

def message_for(website: Website):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_for(website)}
    ]

# Caching
_summary_cache = {}


# Summarization
def summarize(url: str, model: str = "llama3.2") -> str:
    if url in _summary_cache:
        logger.info(f"Cache hit for {url}")
        return _summary_cache[url]

    website = Website(url)
    if not website.text:
        return f"Could not fetch or parse website: {url}"

    try:
        if USE_OPENAI:
            openai = OpenAI(base_url='http://localhost:11434/v1', api_key=API_KEY)
            response = openai.chat.completions.create(
                model=model,
                messages=message_for(website)
            )
            # print(response.choices[0].message.content)
            summary = response.choices[0].message.content
        else:
            response = ollama.chat(
                model=model,
                messages=message_for(website)
            )
            # print(response["message"]["content"])
            summary = response["message"]["content"]
        
        _summary_cache[url] = summary
        return summary
    except Exception as e:
        logger.error(f"Error calling model {model}: {e}")
        return f"Error summarizing {url}"


# Display
def display_summary(url: str, model: str = "llama3.2"):
    summary = summarize(url, model)
    console = Console()
    console.print(Markdown(summary))


# Main
if __name__ == "__main__":
    url = input("Enter URL: ").strip()
    if url:
        display_summary(url)
    else:
        print("No URL provided.")