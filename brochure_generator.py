import os
import logging
import requests
from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv
from bs4 import BeautifulSoup
# from rich.console import Console
# from rich.markdown import Markdown
import ollama
from openai import OpenAI
import json
import sys

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
MODEL = os.getenv("OPENAI_MODEL", "llama3.2")

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
        self.links = ""

        try:
            response = session.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return

        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string.strip() if soup.title else "No title found"
        self.text = self.extract_text(soup, max_chars)
        self.links = self.extract_links(soup)

    def extract_text(self, soup, max_chars):
        if soup.body:
            # Remove irrelevant elements
            for tag in soup(["script", "style", "img", "input", "nav", "footer", "aside", "header"]):
                tag.decompose()
            text = soup.body.get_text(separator="\n", strip=True)
            
            # Truncate if too long
            if len(text) > max_chars:
                logger.warning(f"Text too long ({len(text)} chars). Truncating.")
                text = text[:max_chars]
            return text
        return ""

        
    def extract_links(self, soup):
        links = [link.get('href') for link in soup.find_all('a')]
        return [link for link in links if link and 'http' in link]
    
    def get_contents(self):
        return f"Webpage title:\n{self.title}\nWebpage contents:\n{self.text}\n\n"
      
class LLM_Client:
    def __init__(self, model=MODEL):
        self.model = model

    def get_relevant_links(self, website):
        link_system_prompt = """
        You are given a list of links from a company website.
        Select only relevant links for a brochure (About, Company, Careers, Products, Contact).
        Exclude login, terms, privacy, and emails.

        ### **Instructions**
        - Return **only valid JSON**.
        - **Do not** include explanations, comments, or Markdown.
        - Example output:
        {
            "links": [
                {"type": "about", "url": "https://company.com/about"},
                {"type": "contact", "url": "https://company.com/contact"},
                {"type": "product", "url": "https://company.com/products"}
            ]
        }
        """

        user_prompt = f"""
        Here is the list of links on the website of {website.url}:
        Please identify the relevant web links for a company brochure. Respond in JSON format.
        Do not include login, terms of service, privacy, or email links.
        Links (some might be relative links):
        {', '.join(website.links)}
        """
        # print(USE_OPENAI)
        # print(type(USE_OPENAI))
        if USE_OPENAI:
            openai = OpenAI(base_url='http://localhost:11434/v1', api_key=API_KEY)
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": link_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            return json.loads(response.choices[0].message.content.strip())
        else:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": link_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return json.loads(response["message"]["content"].strip())
            
    def generate_brochure(self, company_name, content, language):
        system_prompt = """
        You are an assistant that analyzes the contents of several relevant pages from a company website \
        and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
        Include details of company culture, customers and careers/jobs if you have the information.
        """

        user_prompt = f"""
        Create a short brochure for '{company_name}' using the following content:
        {content}
        Respond in {language} only, and format your response correctly in Markdown.
        Do NOT escape characters or return extra backslashes.
        """

        if USE_OPENAI:
            openai = OpenAI(base_url='http://localhost:11434/v1', api_key=API_KEY)
            response_stream = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
            
            for chunk in response_stream:
                text = chunk.choices[0].delta.content or ''
                # response += text
                for ch in text:
                    sys.stdout.write(ch)
                    sys.stdout.flush()
            
        else:
            response_stream = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )

            for chunk in response_stream:
                text = chunk["message"]["content"]
                # response += text
                for ch in text:
                    sys.stdout.write(ch)
                    sys.stdout.flush()

        
class BrochureGenerator:
    def __init__(self, company, url, language='English'):
        self.company = company
        self.url = url
        self.language = language
        
        self.website = Website(self.url)
        self.llm_client = LLM_Client()

    def generate(self):
        content = self.website.get_contents()
        links = self.llm_client.get_relevant_links(self.website)
        print(type(links))
        print(links)
        for link in links['links']:
            linked_webpage = Website(link['url'])
            content += f"\n\n{link['type']}:\n"
            content += linked_webpage.get_contents()

        self.llm_client.generate_brochure(self.company, content, self.language)


def main():
    # company = input("Enter Company's name: ")
    # url = input("Enter Company's website: ")
    company = 'Tour Eiffel'
    url = 'https://www.toureiffel.paris'
    language = 'English'

    generator = BrochureGenerator(company, url, language)
    generator.generate()


if __name__ == "__main__":
    main()