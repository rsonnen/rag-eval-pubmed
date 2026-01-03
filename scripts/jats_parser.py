"""Parse JATS XML documents from PubMed Central.

Extracts structured content from NISO JATS (Journal Article Tag Suite) XML
for LLM-based relevance evaluation. Handles the common elements found in
PMC open access articles.

JATS specification: https://jats.nlm.nih.gov/
"""

import logging
from dataclasses import dataclass
from typing import Any

from lxml import etree

logger = logging.getLogger(__name__)


class JATSParseError(Exception):
    """Raised when JATS XML parsing fails to extract required fields."""

    pass


@dataclass
class ParsedArticle:
    """Structured content extracted from a JATS XML article."""

    pmcid: str
    title: str
    abstract: str
    authors: list[str]
    keywords: list[str]
    subjects: list[str]
    journal: str
    pub_date: str
    body_text: str
    license_type: str | None


def _get_text(element: etree._Element | None) -> str:
    """Extract all text content from an element, including nested elements."""
    if element is None:
        return ""
    # itertext() yields strings, join them
    text_parts: list[str] = []
    for text in element.itertext():
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts).strip()


def _find_text(root: etree._Element, xpath: str) -> str:
    """Find an element by xpath and extract its text content."""
    results = root.xpath(xpath)
    if results and isinstance(results, list) and len(results) > 0:
        first = results[0]
        if isinstance(first, etree._Element):
            return _get_text(first)
    return ""


def _extract_title(root: etree._Element) -> str:
    """Extract the article title from front matter."""
    return _find_text(root, ".//front//article-title")


def _extract_abstract(root: etree._Element) -> str:
    """Extract the abstract text.

    Handles both simple abstracts and structured abstracts with sections.
    """
    abstract_elem = root.find(".//front//abstract")
    if abstract_elem is None:
        return ""

    parts: list[str] = []
    for sec in abstract_elem.findall(".//sec"):
        title = _get_text(sec.find("title"))
        text = " ".join(_get_text(p) for p in sec.findall("p"))
        if title:
            parts.append(f"{title}: {text}")
        elif text:
            parts.append(text)

    if parts:
        return " ".join(parts)

    paragraphs = abstract_elem.findall(".//p")
    return " ".join(_get_text(p) for p in paragraphs)


def _extract_authors(root: etree._Element) -> list[str]:
    """Extract author names from the contributor group."""
    authors: list[str] = []
    results = root.xpath(".//front//contrib[@contrib-type='author']")
    if not isinstance(results, list):
        return authors
    for item in results:
        if not isinstance(item, etree._Element):
            continue
        contrib = item
        name_elem = contrib.find("name")
        if name_elem is not None:
            surname = _get_text(name_elem.find("surname"))
            given = _get_text(name_elem.find("given-names"))
            if surname:
                full_name = f"{given} {surname}".strip() if given else surname
                authors.append(full_name)
    return authors


def _extract_keywords(root: etree._Element) -> list[str]:
    """Extract keywords from the article metadata."""
    keywords: list[str] = []
    results = root.xpath(".//front//kwd")
    if not isinstance(results, list):
        return keywords
    for item in results:
        if isinstance(item, etree._Element):
            text = _get_text(item)
            if text:
                keywords.append(text)
    return keywords


def _extract_subjects(root: etree._Element) -> list[str]:
    """Extract subject categories from the article metadata."""
    subjects: list[str] = []
    results = root.xpath(".//front//subject")
    if not isinstance(results, list):
        return subjects
    for item in results:
        if isinstance(item, etree._Element):
            text = _get_text(item)
            if text:
                subjects.append(text)
    return subjects


def _extract_journal(root: etree._Element) -> str:
    """Extract the journal title."""
    journal = _find_text(root, ".//front//journal-title")
    if not journal:
        journal = _find_text(root, ".//front//journal-id")
    return journal


def _extract_pub_date(root: etree._Element) -> str:
    """Extract the publication date in YYYY-MM-DD or YYYY format."""
    pub_date = root.find(".//front//pub-date[@pub-type='epub']")
    if pub_date is None:
        pub_date = root.find(".//front//pub-date[@pub-type='ppub']")
    if pub_date is None:
        pub_date = root.find(".//front//pub-date")

    if pub_date is None:
        return ""

    year = _get_text(pub_date.find("year"))
    month = _get_text(pub_date.find("month"))
    day = _get_text(pub_date.find("day"))

    if year:
        if month and day:
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        if month:
            return f"{year}-{month.zfill(2)}"
        return year
    return ""


def _extract_body_text(root: etree._Element, max_length: int = 8000) -> str:
    """Extract body text for evaluation, up to max_length characters.

    Prioritizes introduction, methods, results sections in that order.
    Skips reference lists and supplementary materials.
    """
    body = root.find(".//body")
    if body is None:
        return ""

    sections: list[str] = []
    total_length = 0

    for sec in body.findall(".//sec"):
        sec_id = sec.get("id", "").lower()
        sec_type = sec.get("sec-type", "").lower()

        if "ref" in sec_id or "supplement" in sec_id or "appendix" in sec_id:
            continue
        if "ref" in sec_type or "supplement" in sec_type:
            continue

        title_elem = sec.find("title")
        title = _get_text(title_elem) if title_elem is not None else ""

        paragraphs: list[str] = []
        for p in sec.findall("p"):
            p_text = _get_text(p)
            if p_text:
                paragraphs.append(p_text)

        if paragraphs:
            section_text = f"{title}:\n" if title else ""
            section_text += " ".join(paragraphs)

            if total_length + len(section_text) > max_length:
                remaining = max_length - total_length
                if remaining > 100:
                    sections.append(section_text[:remaining] + "...")
                break

            sections.append(section_text)
            total_length += len(section_text) + 2

    return "\n\n".join(sections)


def _extract_license(root: etree._Element) -> str | None:
    """Extract the license type from the article metadata."""
    license_elem = root.find(".//front//license")
    if license_elem is None:
        return None

    license_type = license_elem.get("license-type")
    if license_type:
        return license_type

    href = license_elem.get("{http://www.w3.org/1999/xlink}href", "")
    if "creativecommons.org" in href:
        if "/by-nc-nd/" in href:
            return "CC BY-NC-ND"
        if "/by-nc-sa/" in href:
            return "CC BY-NC-SA"
        if "/by-nc/" in href:
            return "CC BY-NC"
        if "/by-nd/" in href:
            return "CC BY-ND"
        if "/by-sa/" in href:
            return "CC BY-SA"
        if "/by/" in href:
            return "CC BY"
        if "/zero/" in href or "publicdomain" in href:
            return "CC0"

    license_text = _get_text(license_elem)
    return license_text[:50] if license_text else None


def _extract_pmcid(root: etree._Element) -> str:
    """Extract the PMC ID from article metadata.

    PMC IDs appear with pub-id-type='pmcid' and include the 'PMC' prefix.
    """
    results = root.xpath(".//front//article-id[@pub-id-type='pmcid']")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, etree._Element):
            text = _get_text(first)
            if text:
                return text if text.startswith("PMC") else f"PMC{text}"
    return ""


def _strip_namespaces(root: etree._Element) -> etree._Element:
    """Remove all namespace prefixes from an element tree.

    OAI-PMH responses wrap JATS articles in namespaced envelopes. Stripping
    namespaces allows simple XPath queries without namespace maps.
    """
    for elem in root.iter():
        if isinstance(elem.tag, str) and elem.tag.startswith("{"):
            elem.tag = elem.tag.split("}", 1)[1]
        # Clean namespace-prefixed attributes (like xlink:href -> href)
        attribs_to_update: list[tuple[str, str]] = []
        for attr_name, _attr_value in elem.attrib.items():
            if isinstance(attr_name, str) and attr_name.startswith("{"):
                new_name = attr_name.split("}", 1)[1]
                attribs_to_update.append((attr_name, new_name))
        for old_name, new_name in attribs_to_update:
            elem.attrib[new_name] = elem.attrib.pop(old_name)
    return root


def _extract_article_from_oai(root: etree._Element) -> etree._Element | None:
    """Extract the JATS article element from an OAI-PMH envelope.

    OAI-PMH GetRecord responses have structure:
        <record><header>...</header><metadata><article>...</article></metadata></record>

    This function finds the article element regardless of namespaces.
    """
    # First strip namespaces so we can use simple element names
    root = _strip_namespaces(root)

    # Check if the root is already an article element
    if root.tag == "article":
        return root

    # Look for article inside OAI-PMH metadata envelope
    article = root.find(".//metadata/article")
    if article is not None:
        return article

    # Fallback: search anywhere for article element
    article = root.find(".//article")
    return article


def parse_jats_xml(xml_content: str | bytes) -> ParsedArticle:
    """Parse JATS XML content and extract structured article data.

    Handles both raw JATS XML and OAI-PMH wrapped responses. OAI-PMH responses
    contain a namespace envelope that is automatically stripped.

    Args:
        xml_content: Raw XML content as string or bytes.

    Returns:
        ParsedArticle with extracted fields.

    Raises:
        JATSParseError: If XML parsing fails or required fields are missing.
    """
    if isinstance(xml_content, str):
        xml_content = xml_content.encode("utf-8")

    try:
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        root = etree.fromstring(xml_content, parser=parser)
    except etree.XMLSyntaxError as e:
        raise JATSParseError(f"XML syntax error: {e}") from e

    article = _extract_article_from_oai(root)
    if article is None:
        raise JATSParseError("No <article> element found in XML")

    pmcid = _extract_pmcid(article)
    if not pmcid:
        raise JATSParseError("Failed to extract PMCID from article-id elements")

    title = _extract_title(article)
    if not title:
        raise JATSParseError(f"Failed to extract title for {pmcid}")

    abstract = _extract_abstract(article)
    # Abstract is required for LLM evaluation
    if not abstract:
        raise JATSParseError(f"No abstract found for {pmcid}")

    return ParsedArticle(
        pmcid=pmcid,
        title=title,
        abstract=abstract,
        authors=_extract_authors(article),
        keywords=_extract_keywords(article),
        subjects=_extract_subjects(article),
        journal=_extract_journal(article),
        pub_date=_extract_pub_date(article),
        body_text=_extract_body_text(article),
        license_type=_extract_license(article),
    )


def article_to_evaluation_dict(article: ParsedArticle) -> dict[str, Any]:
    """Convert a ParsedArticle to a dict suitable for LLM evaluation.

    Returns a dict with the fields formatted for the evaluation prompt.
    """
    return {
        "pmcid": article.pmcid,
        "title": article.title,
        "authors": ", ".join(article.authors) if article.authors else "Unknown",
        "journal": article.journal or "Unknown",
        "pub_date": article.pub_date or "Unknown",
        "keywords": ", ".join(article.keywords) if article.keywords else "None",
        "subjects": ", ".join(article.subjects) if article.subjects else "None",
        "abstract": article.abstract or "No abstract available",
        "body_excerpt": article.body_text[:4000] if article.body_text else "",
        "license": article.license_type or "Unknown",
    }
