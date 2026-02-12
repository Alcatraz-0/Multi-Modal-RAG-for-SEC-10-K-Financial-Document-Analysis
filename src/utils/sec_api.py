"""
SEC EDGAR API interface for downloading 10-K filings and XBRL data
"""
import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re


class SECDownloader:
    """Download 10-K filings and XBRL data from SEC EDGAR"""

    def __init__(self, user_agent: str, data_dir: Path):
        """
        Initialize SEC downloader

        Args:
            user_agent: User agent string (must include email per SEC policy)
            data_dir: Directory to save downloaded files
        """
        self.user_agent = user_agent
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        # Use proper browser headers to avoid being blocked
        # Source - https://stackoverflow.com/a (Posted by Sergey K, Retrieved 2025-12-02, License - CC BY-SA 4.0)
        self.session.headers.update({
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': user_agent,
        })

    def download_10k(
        self,
        cik: str,
        ticker: str,
        start_year: int,
        end_year: int,
        prefer_html: bool = True,
        max_filings: Optional[int] = None
    ) -> List[Dict]:
        """
        Download 10-K filings for a company

        Args:
            cik: Central Index Key (CIK) of the company
            ticker: Stock ticker symbol
            start_year: Starting fiscal year
            end_year: Ending fiscal year
            prefer_html: Prefer HTML format over PDF
            max_filings: Maximum number of filings to download

        Returns:
            List of filing metadata dictionaries
        """
        filings = []

        # Search for 10-K filings
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': '10-K',
            'dateb': f'{end_year}1231',
            'owner': 'exclude',
            'count': 100
        }

        try:
            response = self.session.get(search_url, params=params)
            response.raise_for_status()

            # Parse filing links
            soup = BeautifulSoup(response.content, 'html.parser')
            filing_table = soup.find('table', {'class': 'tableFile2'})

            if not filing_table:
                return filings

            rows = filing_table.find_all('tr')[1:]  # Skip header

            for row in rows:
                if max_filings and len(filings) >= max_filings:
                    break

                cols = row.find_all('td')
                if len(cols) < 4:
                    continue

                filing_type = cols[0].text.strip()
                if filing_type != '10-K':
                    continue

                filing_date = cols[3].text.strip()
                filing_year = int(filing_date.split('-')[0])

                if filing_year < start_year or filing_year > end_year:
                    continue

                # Get document link
                doc_link = cols[1].find('a', {'id': 'documentsbutton'})
                if not doc_link:
                    continue

                documents_url = f"https://www.sec.gov{doc_link['href']}"

                # Download filing
                filing_info = self._download_filing(
                    documents_url,
                    ticker,
                    filing_year,
                    prefer_html
                )

                if filing_info:
                    filings.append(filing_info)

                # Rate limiting
                time.sleep(0.1)

        except Exception as e:
            print(f"Error downloading 10-K for {ticker}: {str(e)}")

        return filings

    def _download_filing(
        self,
        documents_url: str,
        ticker: str,
        fiscal_year: int,
        prefer_html: bool
    ) -> Optional[Dict]:
        """Download individual filing document"""

        try:
            response = self.session.get(documents_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'class': 'tableFile'})

            if not table:
                return None

            # Find the main 10-K document (not the XBRL viewer)
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) < 4:
                    continue

                doc_type = cols[3].text.strip()
                description = cols[1].text.strip().lower()

                # Skip XBRL viewers and lookup for actual 10-K HTM/HTML files
                # We want files that are actual 10-K documents, not viewers
                if 'xml' in description or 'instance' in description:
                    continue

                # Look for the actual 10-K document file
                # Usually it's a .htm file with "10-K" in the type column
                if doc_type == '10-K' or (doc_type == '' and '10-k' in description):
                    doc_link = cols[2].find('a')
                    if not doc_link:
                        continue

                    doc_url = f"https://www.sec.gov{doc_link['href']}"

                    # Skip if it's an XBRL viewer page
                    if 'ix?doc=' in doc_url or 'ixviewer' in doc_url:
                        # Extract the actual document from the viewer URL
                        import urllib.parse
                        parsed = urllib.parse.urlparse(doc_url)
                        query = urllib.parse.parse_qs(parsed.query)
                        if 'doc' in query:
                            # Get the actual document path (starts with /)
                            doc_path = query['doc'][0]
                            # The doc_path already contains the full path from root
                            doc_url = f"https://www.sec.gov{doc_path}"

                    file_format = 'html' if doc_url.endswith(('.htm', '.html')) else 'pdf'

                    # Download document
                    print(f"  Downloading: {doc_url}")
                    doc_response = self.session.get(doc_url)
                    doc_response.raise_for_status()

                    # Check if we got actual content (not a viewer page)
                    content = doc_response.text if isinstance(doc_response.content, bytes) else doc_response.content
                    if len(content) < 10000:  # Viewer pages are small
                        print(f"  Warning: Downloaded file seems small ({len(content)} bytes), might be viewer page")
                        continue

                    # Save to file
                    company_dir = self.data_dir / ticker
                    company_dir.mkdir(parents=True, exist_ok=True)

                    file_ext = 'html' if file_format == 'html' else 'pdf'
                    output_file = company_dir / f"{ticker}_{fiscal_year}_10K.{file_ext}"

                    with open(output_file, 'wb') as f:
                        f.write(doc_response.content)

                    print(f"  ✓ Saved {len(doc_response.content)} bytes to {output_file.name}")

                    return {
                        'fiscal_year': fiscal_year,
                        'format': file_format,
                        'path': str(output_file),
                        'url': doc_url
                    }

        except Exception as e:
            print(f"Error downloading filing document: {str(e)}")
            return None

    def download_xbrl(
        self,
        cik: str,
        ticker: str,
        start_year: int,
        end_year: int
    ) -> List[Dict]:
        """
        Download XBRL financial data

        Args:
            cik: Central Index Key
            ticker: Stock ticker
            start_year: Starting fiscal year
            end_year: Ending fiscal year

        Returns:
            List of XBRL file metadata
        """
        xbrl_files = []

        # XBRL data is available from SEC's Financial Statement Data Sets
        # This is a simplified implementation
        # In practice, you would use SEC's bulk data download or API

        print(f"XBRL download for {ticker} - Implementation needed")

        return xbrl_files
