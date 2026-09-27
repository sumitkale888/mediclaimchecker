"""PubMed E-utilities service for dynamic research paper retrieval."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import LLMServiceError, PubMedServiceError

logger = logging.getLogger(__name__)

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_DATABASE = "pubmed"





class PubMedService:
    """Service for searching and retrieving PubMed articles via NCBI E-utilities."""

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        tool: str | None = None,
    ) -> None:
        self.email = (email or settings.ncbi_email).strip()
        self.api_key = (api_key or settings.ncbi_api_key).strip()
        self.tool = (tool or settings.ncbi_tool).strip()
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def _validate_config(self) -> None:
        if not self.email:
            raise PubMedServiceError(
                "NCBI_EMAIL is not configured. Set it in the .env file before using PubMed search."
            )
        if not self.api_key:
            raise PubMedServiceError(
                "NCBI_API_KEY is not configured. Set it in the .env file before using PubMed search."
            )

    def _build_params(self, params: dict[str, Any]) -> dict[str, Any]:
        base_params = {
            "email": self.email,
            "tool": self.tool,
            "api_key": self.api_key,
        }
        base_params.update(params)
        return base_params

    def search_pubmed(self, query: str, max_results: int = 5) -> list[str]:
        """
        Search PubMed using NCBI ESearch and return PMIDs.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of PubMed IDs (PMIDs)

        Raises:
            PubMedServiceError: If search fails or no results found
        """
        self._validate_config()
        client = self._get_client()

        logger.info("PubMed search started for query: %s", query[:100])

        params = self._build_params({
            "db": NCBI_DATABASE,
            "term": query,
            "retmax": str(max_results),
            "retmode": "xml",
        })

        try:
            response = client.get(f"{NCBI_BASE_URL}/esearch.fcgi", params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.exception("PubMed ESearch request failed")
            raise PubMedServiceError("PubMed search request failed. Please try again later.") from exc
        except httpx.TimeoutException as exc:
            logger.exception("PubMed API timeout")
            raise PubMedServiceError("PubMed API request timed out. Please try again later.") from exc
        except PubMedServiceError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during PubMed search")
            raise PubMedServiceError("PubMed search failed unexpectedly.") from exc

        try:
            root = ET.fromstring(response.text)
            id_list = root.find(".//IdList")

            if id_list is None:
                logger.warning("No PubMed results found for query: %s", query[:100])
                return []

            pmids = [id_elem.text for id_elem in id_list.findall("Id") if id_elem.text]

            logger.info("PubMed returned %d PMIDs", len(pmids))
            return pmids

        except ET.ParseError as exc:
            logger.exception("Failed to parse PubMed XML response")
            raise PubMedServiceError("Failed to parse PubMed response.") from exc

    def fetch_pubmed_articles(self, pmids: list[str]) -> list[dict[str, Any]]:
        """
        Fetch article details for given PMIDs using NCBI EFetch.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of article dictionaries with metadata

        Raises:
            PubMedServiceError: If fetch fails
        """
        if not pmids:
            return []

        self._validate_config()
        client = self._get_client()

        logger.info("Fetching %d PubMed articles", len(pmids))

        params = self._build_params({
            "db": NCBI_DATABASE,
            "id": ",".join(pmids),
            "retmode": "xml",
        })

        try:
            response = client.get(f"{NCBI_BASE_URL}/efetch.fcgi", params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.exception("PubMed EFetch request failed")
            raise PubMedServiceError("Failed to fetch PubMed articles. Please try again later.") from exc
        except httpx.TimeoutException as exc:
            logger.exception("PubMed API timeout")
            raise PubMedServiceError("PubMed API request timed out. Please try again later.") from exc
        except Exception as exc:
            logger.exception("Unexpected error during PubMed fetch")
            raise PubMedServiceError("Failed to fetch PubMed articles unexpectedly.") from exc

        try:
            return self._parse_pubmed_xml(response.text)
        except ET.ParseError as exc:
            logger.exception("Failed to parse PubMed XML response")
            raise PubMedServiceError("Failed to parse PubMed article data.") from exc

    def _parse_pubmed_xml(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse PubMed XML response and extract article metadata."""
        root = ET.fromstring(xml_text)
        articles = []

        for pubmed_article in root.findall(".//PubmedArticle"):
            try:
                article = self._extract_article_data(pubmed_article)
                if article:
                    articles.append(article)
            except Exception as exc:
                logger.warning("Failed to parse individual PubMed article: %s", exc)
                continue

        logger.info("Successfully parsed %d articles", len(articles))
        return articles

    def _extract_article_data(self, pubmed_article: ET.Element) -> dict[str, Any] | None:
        """Extract article data from a single PubmedArticle element."""
        pmid_elem = pubmed_article.find(".//PMID")
        if pmid_elem is None or pmid_elem.text is None:
            return None

        pmid = pmid_elem.text

        article = pubmed_article.find(".//Article")
        if article is None:
            return None

        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        abstract_elem = article.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else None

        authors = []
        author_list = article.find(".//AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                initials = author.find("Initials")

                if last_name is not None and last_name.text:
                    author_name = last_name.text
                    if fore_name is not None and fore_name.text:
                        author_name = f"{fore_name.text} {author_name}"
                    elif initials is not None and initials.text:
                        author_name = f"{initials.text} {author_name}"
                    authors.append(author_name)

        journal_elem = pubmed_article.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None and journal_elem.text else None

        year_elem = pubmed_article.find(".//PubDate/Year")
        year = None
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text)
            except ValueError:
                pass

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "publication_year": year,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
