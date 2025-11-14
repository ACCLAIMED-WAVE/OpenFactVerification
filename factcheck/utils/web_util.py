import requests
import backoff
import time
import bs4
import asyncio
from httpx import AsyncHTTPTransport
from httpx._client import AsyncClient


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
# mobile user-agent
MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"
headers = {"User-Agent": USER_AGENT}


def is_tag_visible(element: bs4.element) -> bool:
    """Determines if an HTML element is visible.

    Args:
        element: A BeautifulSoup element to check the visibility of.
    returns:
        Whether the element is visible.
    """
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ] or isinstance(element, bs4.element.Comment):
        return False
    return True


transport = AsyncHTTPTransport(retries=3)


async def httpx_get(url: str, headers: dict):
    try:
        async with AsyncClient(transport=transport) as client:
            response = await client.get(url, headers=headers, timeout=3)
            response = response if response.status_code == 200 else None
            if not response:
                return False, None
            else:
                return True, response
    except Exception as e:  # noqa: F841
        return False, None


async def httpx_bind_key(url: str, headers: dict, key: str = ""):
    flag, response = await httpx_get(url, headers)
    return flag, response, url, key


def crawl_web(query_url_dict: dict):
    tasks = list()
    for query, urls in query_url_dict.items():
        for url in urls:
            task = httpx_bind_key(url=url, headers=headers, key=query)
            tasks.append(task)
    asyncio.set_event_loop(asyncio.SelectorEventLoop())
    loop = asyncio.get_event_loop()
    responses = loop.run_until_complete(asyncio.gather(*tasks))
    return responses


# @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, requests.exceptions.Timeout), max_tries=1,max_time=3)
def common_web_request(url: str, query: str = None, timeout: int = 3):
    from factcheck.utils.logger import CustomLogger
    logger = CustomLogger(__name__).getlog()
    
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        logger.debug(f"[common_web_request] Request to {url}: status={resp.status_code}, size={len(resp.text)} bytes")
        if resp.status_code != 200:
            logger.warning(f"[common_web_request] Non-200 status code for {url}: {resp.status_code}")
        if query:
            return resp, query
        else:
            return resp
    except requests.exceptions.RequestException as e:
        logger.error(f"[common_web_request] Request failed for {url}: {e}")
        raise


def parse_response(response: requests.Response, url: str, query: str = None):
    html_content = response.text
    url = url
    try:
        soup = bs4.BeautifulSoup(html_content, "html.parser")
        texts = soup.findAll(text=True)
        # Filter out invisible text from the page.
        visible_text = filter(is_tag_visible, texts)
    except Exception as _:  # noqa: F841
        return None, url, query

    # Returns all the text concatenated as a string.
    web_text = " ".join(t.strip() for t in visible_text).strip()
    # Clean up spacing.
    web_text = " ".join(web_text.split())
    return web_text, url, query


def scrape_url(url: str, timeout: float = 3):
    """Scrapes a URL for all text information.

    Args:
        url: URL of webpage to scrape.
        timeout: Timeout of the requests call.
    Returns:
        web_text: The visible text of the scraped URL.
        url: URL input.
    """
    # Scrape the URL
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as _:  # noqa: F841
        return None, url

    # Extract out all text from the tags
    try:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        texts = soup.findAll(text=True)
        # Filter out invisible text from the page.
        visible_text = filter(is_tag_visible, texts)
    except Exception as _:  # noqa: F841
        return None, url

    # Returns all the text concatenated as a string.
    web_text = " ".join(t.strip() for t in visible_text).strip()
    # Clean up spacing.
    web_text = " ".join(web_text.split())
    return web_text, url


def crawl_google_web(response, top_k: int = 10, page_index: int = None):
    from factcheck.utils.logger import CustomLogger
    logger = CustomLogger(__name__).getlog()
    
    if not response or not hasattr(response, 'text'):
        logger.error(f"[crawl_google_web] Invalid response object received")
        return []
    
    response_text = response.text if hasattr(response, 'text') else str(response)
    if not response_text or len(response_text) < 100:
        logger.warning(f"[crawl_google_web] Response is empty or too short (length: {len(response_text) if response_text else 0})")
        return []
    
    soup = bs4.BeautifulSoup(response_text, "html.parser")
    
    all_links = soup.find_all("a", {"href": True})
    logger.debug(f"[crawl_google_web] Found {len(all_links)} total <a> tags with href attributes")
    
    valid_node_list = list()
    for node in all_links:
        if node.findChildren("h3"):
            valid_node_list.append(node)
    
    logger.debug(f"[crawl_google_web] Found {len(valid_node_list)} links with <h3> children (Google search result pattern)")
    
    result_urls = list()
    for node in valid_node_list:
        href = node.get("href")
        if href:
            result_urls.append(href)
    
    if len(result_urls) == 0:
        timestamp = int(time.time() * 1000000)
        index_str = f"_{page_index}" if page_index is not None else ""
        filename = f"/tmp/google_result_page_{timestamp}{index_str}.html"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response_text)
            logger.warning(f"[crawl_google_web] No URLs extracted! Saved full HTML to {filename}")
        except Exception as e:
            logger.error(f"[crawl_google_web] Failed to save HTML to {filename}: {e}")
        logger.warning(f"[crawl_google_web] Response length: {len(response_text)}, Response preview: {response_text[:500]}")
        logger.warning(f"[crawl_google_web] This likely means Google's HTML structure has changed or the request was blocked")
        if "captcha" in response_text.lower() or "unusual traffic" in response_text.lower():
            logger.error(f"[crawl_google_web] Google may have blocked the request (CAPTCHA detected)")
    
    return result_urls[:top_k]
