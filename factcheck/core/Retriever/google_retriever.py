from concurrent.futures import ThreadPoolExecutor
from factcheck.utils.web_util import common_web_request, crawl_google_web
from .base import BaseRetriever
from factcheck.utils.logger import CustomLogger

logger = CustomLogger(__name__).getlog()


class GoogleEvidenceRetriever(BaseRetriever):
    def __init__(self, llm_client, api_config: dict = None) -> None:
        super(GoogleEvidenceRetriever, self).__init__(llm_client, api_config)
        self.num_web_pages = 10

    def _get_query_urls(self, questions: list[str]):
        logger.info(f"[GoogleRetriever] Starting URL extraction for {len(questions)} queries: {questions}")
        all_request_url_dict = dict()
        for query in questions:
            query = query.replace(" ", "+")
            curr_query_list = all_request_url_dict.get(query, [])
            for page in range(0, self.num_web_pages, 10):
                # here page is google search's bottom page meaning, click 2 -> start=10
                # url = "https://www.google.com/search?q={}&start={}".format(query, page)
                url = "https://www.google.com/search?q={}&lr=lang_{}&hl={}&start={}".format(query, self.lang, self.lang, page)
                curr_query_list.append(url)
                all_request_url_dict[query] = curr_query_list
        logger.info(f"[GoogleRetriever] Generated {sum(len(urls) for urls in all_request_url_dict.values())} Google search URLs to fetch")

        crawled_all_page_urls_dict = dict()
        page_index = 0
        with ThreadPoolExecutor(max_workers=len(all_request_url_dict.values())) as executor:
            futures = list()
            for query, urls in all_request_url_dict.items():
                for url in urls:
                    future = executor.submit(common_web_request, url, query)
                    futures.append(future)
            for future in futures:
                response, query = future.result()
                logger.info(f"[GoogleRetriever] Fetched Google search page for query '{query}': status={response.status_code}, content_length={len(response.text)}")
                extracted_urls = crawl_google_web(response, page_index=page_index)
                logger.info(f"[GoogleRetriever] Extracted {len(extracted_urls)} URLs from Google search page for query '{query}': {extracted_urls[:3]}")
                content_list = crawled_all_page_urls_dict.get(query, [])
                content_list.extend(extracted_urls)
                crawled_all_page_urls_dict[query] = content_list
                page_index += 1
        for query, urls in crawled_all_page_urls_dict.items():
            # urls = sorted(list(set(urls)))
            original_count = len(urls)
            crawled_all_page_urls_dict[query] = urls[: self.max_search_result_per_query]
            logger.info(f"[GoogleRetriever] Query '{query}': Found {original_count} total URLs, keeping top {len(crawled_all_page_urls_dict[query])}")
        total_urls = sum(len(urls) for urls in crawled_all_page_urls_dict.values())
        logger.info(f"[GoogleRetriever] URL extraction complete. Total URLs to crawl: {total_urls}")
        return crawled_all_page_urls_dict
