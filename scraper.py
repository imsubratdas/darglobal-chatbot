
import os
import re
import json
import gzip
import time
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

from io import BytesIO
from datetime import datetime, timezone
from collections import deque
from urllib.parse import (
    urljoin,
    urlparse,
    urldefrag,
    urlunparse
)
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# ==========================================
# 1. CONFIGURATION
# ==========================================

USER_AGENT = (
    "darglobal-bot/1.0 "
    "(contact: your-email@example.com)"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml"
}

MAX_PAGES_PER_SITE = 40
DELAY_SECONDS = 2
REQUEST_TIMEOUT = 25

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR_SETTING = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR = (DATA_DIR_SETTING if DATA_DIR_SETTING.is_absolute() else BASE_DIR / DATA_DIR_SETTING).resolve()
OUTPUT_FILE = str(DATA_DIR / "pages.json")

SKIP_EXT = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif",
    ".svg", ".webp", ".zip", ".mp4",
    ".css", ".js", ".woff", ".woff2"
)

SITES = [
    {
        "company": "DarGlobal",
        "start": "https://darglobal.co.uk/",
        "sitemaps": [
            "https://darglobal.co.uk/sitemap.xml"
        ]
    },
    {
        "company": "Wasalt",
        "start": "https://wasalt.sa/en/",
        "sitemaps": [
            "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/rental_srp_sitemap_en_sa.xml.gz"
        ]
    }
]


# ==========================================
# 2. HTTP SESSION
# ==========================================

session = requests.Session()
session.headers.update(HEADERS)


# ==========================================
# 3. URL UTILITIES
# ==========================================

def clean_host(url):
    host = urlparse(url).hostname or ""
    return host.lower().removeprefix("www.")


def normalize_url(url):
    """
    Remove fragments and unnecessary trailing slashes.
    Keep meaningful query parameters out of the crawl.
    """

    url = urldefrag(url)[0]
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.query:
        return None

    path = parsed.path or "/"

    if path != "/":
        path = path.rstrip("/")

    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        "",
        "",
        ""
    ))


# ==========================================
# 4. ROBOTS.TXT HANDLING
# ==========================================

robots_cache = {}


def get_robots(base_url):

    parsed = urlparse(base_url)

    robots_url = (
        f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    )

    if robots_url in robots_cache:
        return robots_cache[robots_url]

    rp = RobotFileParser()
    rp.set_url(robots_url)

    try:

        response = session.get(
            robots_url,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            print(
                f"[ROBOTS ERROR] {robots_url}: "
                f"HTTP {response.status_code}"
            )
            robots_cache[robots_url] = None
            return None

        rp.parse(response.text.splitlines())

        robots_cache[robots_url] = rp

        print(f"[ROBOTS LOADED] {robots_url}")

        return rp

    except requests.RequestException as error:

        print(f"[ROBOTS ERROR] {error}")

        robots_cache[robots_url] = None

        return None


def is_allowed(url):

    robots = get_robots(url)

    # Fail closed when robots rules cannot be checked.
    if robots is None:
        return False

    allowed = robots.can_fetch(USER_AGENT, url)

    if not allowed:
        print(f"[ROBOTS BLOCKED] {url}")

    return allowed


# ==========================================
# 5. DOWNLOAD AND PARSE SITEMAPS
# ==========================================

def extract_sitemap(url):

    if not is_allowed(url):
        print(f"[SITEMAP BLOCKED] {url}")
        return []

    try:

        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        content = response.content

        # Handle gzip-compressed XML.
        if content.startswith(b"\x1f\x8b"):
            content = gzip.decompress(content)

        root = ET.fromstring(content)

        urls = []

        for element in root.iter():

            if element.tag.endswith("loc") and element.text:

                loc = element.text.strip()

                urls.append(loc)

        return urls

    except Exception as error:

        print(f"[SITEMAP ERROR] {url}: {error}")

        return []


def discover_sitemap_urls(site):

    print(
        f"\nDiscovering sitemap URLs: "
        f"{site['company']}"
    )

    discovered = set()
    sitemap_queue = deque(site["sitemaps"])
    visited_sitemaps = set()

    while sitemap_queue:

        sitemap_url = sitemap_queue.popleft()

        if sitemap_url in visited_sitemaps:
            continue

        visited_sitemaps.add(sitemap_url)

        entries = extract_sitemap(sitemap_url)

        for entry in entries:

            # Sitemap indexes can contain other sitemaps.
            if entry.lower().endswith((".xml", ".xml.gz")):

                if entry not in visited_sitemaps:
                    sitemap_queue.append(entry)

            else:

                normalized = normalize_url(entry)

                if normalized:
                    discovered.add(normalized)

    print(
        f"[SITEMAP] Discovered "
        f"{len(discovered)} page URLs"
    )

    return list(discovered)


# ==========================================
# 6. HTML EXTRACTION
# ==========================================

def extract(html):

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "form",
        "nav",
        "footer"
    ]):

        tag.decompose()

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title else ""
    )

    description = ""

    meta = soup.find(
        "meta",
        attrs={"name": "description"}
    )

    if meta:
        description = meta.get("content", "").strip()

    seen = set()
    parts = []

    for element in soup.find_all([
        "h1", "h2", "h3", "h4",
        "p", "li", "td"
    ]):

        text = " ".join(
            element.get_text(" ", strip=True).split()
        )

        if len(text) > 25 and text not in seen:

            seen.add(text)
            parts.append(text)

    return {
        "title": title,
        "description": description,
        "text": " ".join(parts)
    }


# ==========================================
# 7. PAGE CRAWLER
# ==========================================

def crawl_page(url, company, host):

    if clean_host(url) != host:
        return None, []

    if not is_allowed(url):
        return None, []

    try:

        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            print(
                f"[HTTP {response.status_code}] {url}"
            )
            return None, []

        # Do not follow a redirect outside the original site.
        if clean_host(response.url) != host:
            print(f"[EXTERNAL REDIRECT] {url}")
            return None, []

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" not in content_type:
            return None, []

        extracted = extract(response.text)

        record = None

        if len(extracted["text"]) > 200:

            record = {
                "company": company,
                "url": url,
                "title": extracted["title"],
                "description": extracted["description"],
                "text": extracted["text"],
                "scraped_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

        # Discover internal links.
        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        links = set()

        for anchor in soup.find_all("a", href=True):

            raw_link = anchor["href"].strip()

            absolute = urljoin(url, raw_link)

            normalized = normalize_url(absolute)

            if not normalized:
                continue

            if clean_host(normalized) != host:
                continue

            if normalized.lower().endswith(SKIP_EXT):
                continue

            links.add(normalized)

        return record, list(links)

    except requests.RequestException as error:

        print(f"[REQUEST ERROR] {url}: {error}")

        return None, []


# ==========================================
# 8. COMPLETE SITE CRAWLER
# ==========================================

def crawl(site):

    company = site["company"]
    start = normalize_url(site["start"])
    host = clean_host(start)

    print("\n================================")
    print(f"CRAWLING: {company}")
    print("================================")

    # Check the homepage permission first.
    if not is_allowed(start):

        print("[STOP] Homepage not permitted.")
        return []

    # Discover URLs through sitemaps.
    sitemap_urls = discover_sitemap_urls(site)

    # Start with the homepage and sitemap pages.
    queue = deque([start])

    queue.extend(sitemap_urls)

    visited = set()
    results = []

    while queue and len(results) < MAX_PAGES_PER_SITE:

        url = queue.popleft()

        if url in visited:
            continue

        visited.add(url)

        if clean_host(url) != host:
            continue

        record, new_links = crawl_page(
            url,
            company,
            host
        )

        if record:

            results.append(record)

            print(
                f"[{len(results)}/{MAX_PAGES_PER_SITE}] "
                f"{record['title']} "
                f"({len(record['text'])} chars)"
            )

        # Add newly discovered links.
        for link in new_links:

            if link not in visited:
                queue.append(link)

        time.sleep(DELAY_SECONDS)

    print(
        f"\n[COMPLETE] {company}: "
        f"{len(results)} pages collected"
    )

    return results


# ==========================================
# 9. SAVE DATA
# ==========================================

def save_data(records):

    os.makedirs("data", exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n================================")
    print("DATA COLLECTION SUMMARY")
    print("================================")

    print("Total pages:", len(records))

    for company in ("DarGlobal", "Wasalt"):

        count = sum(
            1 for item in records
            if item["company"] == company
        )

        print(f"{company}: {count} pages")

    print("Saved to:", OUTPUT_FILE)


# ==========================================
# 10. MAIN
# ==========================================

if __name__ == "__main__":

    all_pages = []

    for site in SITES:

        try:

            pages = crawl(site)

            all_pages.extend(pages)

        except Exception as error:

            print(
                f"[SITE ERROR] {site['company']}: {error}"
            )

    # Remove duplicate records by URL.
    unique_pages = {}

    for page in all_pages:
        unique_pages[page["url"]] = page

    save_data(list(unique_pages.values()))
