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
    max_results=5,
    topic="news",
    search_depth="advanced",
    include_answer=False,
    include_raw_content=False
)


class SourceAnalysis(BaseModel):

    website_name: str

    article_title: str

    article_url: str

    source_category: Literal[
        "GOVERNMENT",
        "ESTABLISHED_NEWS",
        "FACT_CHECKER",
        "UNKNOWN"
    ]

    evidence_type: Literal[
        "SUPPORTS",
        "CONTRADICTS",
        "MENTIONS_ONLY",
        "IRRELEVANT"
    ]

    evidence: str = Field(
        description="Specific evidence from the article"
    )


class NewsVerification(BaseModel):

    verdict: Literal[
        "SOURCE_FOUND",
        "NO_SOURCE_FOUND",
        "LIKELY_FAKE"
    ]

    explanation: str

    sources: list[SourceAnalysis]

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

    # Generate multiple search queries
    queries = generate_search_queries(title, body)

    # Execute all searches
    results = search_multiple_queries(queries)

    # Remove duplicate articles
    results = remove_duplicate_results(results)

    print(f"\nTotal unique results: {len(results)}")

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

    selected_results = results[:8]

    for result in results:

        formatted_results.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
        })

    search_context = json.dumps(
        formatted_results,
        indent=2,
        ensure_ascii=False
    )

    print("\n========== SEARCH STATISTICS ==========")

    print("Total Tavily results:", len(results))

    print("Results sent to Gemini:", len(formatted_results))

    print(
        "Approximate input characters:",
        len(prompt) if 'prompt' in locals() else len(search_context)
    )

    print("=======================================")

    print(
    "\nSearch context characters:",
    len(search_context)
    )

    prompt = f"""

      You are an expert news verification assistant.

      Your task is to investigate the submitted news
      using the provided internet search results.

      NEWS TITLE:
      {title}

      NEWS BODY:
      {body}

      INTERNET SEARCH RESULTS:
      {search_context}


      INSTRUCTIONS:

      For EVERY relevant search result:

      1. Identify the website name.
      2. Extract the article title.
      3. Extract its URL.
      4. Classify the website.
      5. Determine whether the article supports,
         contradicts, or merely mentions the claim.
      6. Extract the relevant evidence.

      SOURCE CATEGORIES:

      GOVERNMENT:
      Official government websites and government agencies.

      ESTABLISHED_NEWS:
      Recognized news organizations such as Reuters,
      BBC, AP, CNN and similar publishers.

      FACT_CHECKER:
      Organizations specializing in fact-checking.

      UNKNOWN:
      Websites whose credibility cannot be established
      from the available information.

      EVIDENCE TYPES:

      SUPPORTS:
      Provides evidence supporting the news.

      CONTRADICTS:
      Provides evidence against the news.

      MENTIONS_ONLY:
      Mentions the topic without establishing
      whether the claim is true or false.

      IRRELEVANT:
      Does not provide useful evidence.

      IMPORTANT RULES:

      - Never invent websites or URLs.
      - Use only URLs from the search results.
      - Do not assume a source is credible simply
        because it appears in search results.
      - Do not classify a source based only on its domain
        if the publisher cannot be identified.
      - Do not confuse mentioning a claim with confirming it.
      - Do not assume that absence of evidence means fake news.
      - Consider publication dates.
      - Distinguish between partial and complete confirmation.

      VERDICT:

      SOURCE_FOUND:
      At least one source provides meaningful supporting evidence.

      NO_SOURCE_FOUND:
      No meaningful supporting evidence was identified.

      LIKELY_FAKE:
      Reliable evidence directly contradicts important claims.

      Return structured output.
      """

    response = structured_llm.invoke(prompt)

    return response


def generate_search_queries(title, body):

    queries = [

        # Search 1: Exact headline
        f'"{title}"',

        # Search 2: News coverage
        f'{title} news reports latest developments',

        # Search 3: Official confirmation
        f'{title} official government announcement legislation',

        # Search 4: Fact checking
        f'{title} fact check fake or misleading'

    ]

    return queries

def search_multiple_queries(queries):

    all_results = []

    for index, query in enumerate(queries, start=1):

        print(f"\nExecuting search {index}/{len(queries)}")
        print("Query:", query)

        results = tavily.invoke({
            "query": query
        })

        search_results = results.get("results", [])

        all_results.extend(search_results)

    return all_results

def remove_duplicate_results(results):

    unique_results = {}

    for result in results:

        url = result.get("url")

        if url and url not in unique_results:

            unique_results[url] = result

    return list(unique_results.values())





if __name__ == "__main__":

    title = input("\nEnter news title: ")

    print("\nEnter news body:")

    body = input()

    result = verify_news(title, body)

    print("\n========== NEWS VERIFICATION ==========")

    print("\nVerdict:", result.verdict)
    
    print("\nExplanation:")
    print(result.explanation)
    
    print("\nSOURCE ANALYSIS")
    
    for source in result.sources:
    
        print("\n------------------------------")
    
        print("Website:", source.website_name)
    
        print("Article:", source.article_title)
    
        print("URL:", source.article_url)
    
        print("Category:", source.source_category)
    
        print("Evidence Type:", source.evidence_type)
    
        print("Evidence:", source.evidence)
    
        print("------------------------------")

print("\nConfidence:", result.confidence)

print("\n=======================================")
