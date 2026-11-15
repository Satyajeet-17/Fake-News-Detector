import os
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


tavily = TavilySearch(
    max_results=8,
    topic="news",
    search_depth="advanced",
    include_answer=False,
    include_raw_content=True
)

class NewsVerification(BaseModel):

    verdict: Literal[
        "SOURCE_FOUND",
        "NO_SOURCE_FOUND",
        "LIKELY_FAKE"
    ] = Field(
        description="Final verification result"
    )

    explanation: str = Field(
        description="Reason behind the verdict"
    )

    matching_sources: list[str] = Field(
        description="Websites that reported or support the news"
    )

    source_urls: list[str] = Field(
        description="URLs of matching articles"
    )

    contradictory_sources: list[str] = Field(
        description="Websites providing evidence against the news"
    )

    confidence: Literal[
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

structured_llm = llm.with_structured_output(
    NewsVerification
)

def verify_news(title, body):

    print("\nSearching the internet...")

    # Create search query
    search_query = f"""
    Find reliable news reports and fact checks
    about the following claim.

    Title: {title}

    Body: {body}

    Search for:
    1. Exact title matches.
    2. News articles reporting the same event.
    3. Independent reports confirming the claims.
    4. Fact-checking articles.
    5. Credible reports contradicting the claims.

    Focus on reliable news organizations and
    official government sources.
    """

    # Search internet using Tavily
    search_results = tavily.invoke(
        {"query": search_query}
    )

    results = search_results.get("results", [])

    print(f"\nFound {len(results)} search results.")

    # If no results are found
    if not results:

        return {
            "verdict": "NO_SOURCE_FOUND",
            "explanation": (
                "No relevant sources were found "
                "during the internet search."
            ),
            "matching_sources": [],
            "source_urls": [],
            "contradictory_sources": [],
            "confidence": "LOW"
        }

    # Prepare search results for Gemini
    formatted_results = []

    for result in results:

        formatted_results.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
            "raw_content": result.get("raw_content")
        })

    search_context = json.dumps(
        formatted_results,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""

    You are an expert news verification assistant.

    Your task is to verify the submitted news
    using the provided internet search results.

    NEWS TITLE:
    {title}

    NEWS BODY:
    {body}


    INTERNET SEARCH RESULTS:
    {search_context}


    VERIFICATION INSTRUCTIONS:

    1. Examine the submitted title and body.

    2. Compare their factual claims with the
       information in the search results.

    3. Look for websites reporting the same event.

    4. Check whether the sources actually support
       the specific claims in the submitted news.

    5. Identify reliable sources that contradict
       the submitted claims.

    6. Do not assume a news article is authentic
       simply because its title looks similar.

    7. Do not treat search engine snippets as
       conclusive evidence.

    8. Do not invent website names or URLs.

    VERDICT RULES:

    SOURCE_FOUND:
    At least one identifiable source provides
    meaningful evidence supporting the news.

    NO_SOURCE_FOUND:
    No relevant supporting source was found,
    or the available evidence is insufficient.

    LIKELY_FAKE:
    Reliable evidence directly contradicts
    important factual claims in the submitted news.

    IMPORTANT:
    - Absence of evidence is not evidence of falsity.
    - A matching headline alone is insufficient.
    - Consider publication dates.
    - Distinguish partial confirmation from full confirmation.
    - Only return URLs present in the search results.
    - Be conservative when assigning confidence.

    Return the structured verification result.

    """

    response = structured_llm.invoke(prompt)

    return response



if __name__ == "__main__":

    title = input("\nEnter news title: ")

    print("\nEnter news body:")

    body = input()

    result = verify_news(title, body)

    print("\n========== NEWS VERIFICATION ==========")

    print("\nVerdict:", result.verdict)

    print("\nExplanation:")
    print(result.explanation)

    print("\nMatching Sources:")

    for source in result.matching_sources:
        print("-", source)

    print("\nSource URLs:")

    for url in result.source_urls:
        print("-", url)

    print("\nContradictory Sources:")

    for source in result.contradictory_sources:
        print("-", source)

    print("\nConfidence:", result.confidence)

    print("\n=======================================")
    