"""The four specialized agents that make up the research crew.

Each agent has a distinct role, goal and backstory that shapes its behavior:

* **Research Lead** — decomposes the topic into focused sub-questions.
* **Web Researcher** — the only agent with the search tool; gathers sourced
  findings for each sub-question.
* **Analyst** — synthesizes raw findings into structured insight and flags
  contradictions. Also performs the fact-check / review pass.
* **Report Writer** — turns the analysis into a clean, cited markdown report.

Agents are created via :func:`build_agents` so the LLM, tools and tracing
callbacks are injected consistently and are easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass

from crewai import LLM, Agent

from research_crew.tools.search_tool import TavilySearchTool


@dataclass
class CrewAgents:
    """The assembled crew's agents, kept together for the task graph."""

    research_lead: Agent
    web_researcher: Agent
    analyst: Agent
    report_writer: Agent


def build_agents(
    llm: LLM,
    search_tool: TavilySearchTool,
    *,
    verbose: bool = True,
) -> CrewAgents:
    """Construct the four crew agents.

    Args:
        llm: Shared CrewAI LLM instance every agent reasons with.
        search_tool: The web search tool, given only to the Web Researcher.
        verbose: Whether agents narrate their reasoning to the console.

    Returns:
        A :class:`CrewAgents` bundle.
    """
    research_lead = Agent(
        role="Research Lead",
        goal=(
            "Turn the user's topic into 3-5 focused, non-overlapping "
            "sub-questions that together give complete coverage, then set the "
            "research agenda for the crew."
        ),
        backstory=(
            "You are a seasoned research director. You are ruthless about "
            "scope: you break a broad topic into the few questions that truly "
            "matter, avoid redundancy, and phrase each sub-question so it can "
            "be answered with evidence rather than opinion."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=verbose,
        max_iter=10,
    )

    web_researcher = Agent(
        role="Web Researcher",
        goal=(
            "For each sub-question, use the web_search tool to gather the most "
            "relevant, recent and credible sources, and return findings that "
            "always include the source URL."
        ),
        backstory=(
            "You are a meticulous investigative researcher. You never invent "
            "facts: every claim you report is tied to a URL returned by the "
            "search tool. You prefer primary and reputable secondary sources, "
            "and you note when evidence is thin or missing."
        ),
        llm=llm,
        tools=[search_tool],
        allow_delegation=False,
        verbose=verbose,
        max_iter=15,
    )

    analyst = Agent(
        role="Analyst",
        goal=(
            "Synthesize raw findings into structured, decision-useful insight; "
            "explicitly flag where sources conflict; and cut filler."
        ),
        backstory=(
            "You are a sharp analyst who values signal over volume. You group "
            "findings into themes, distinguish well-supported conclusions from "
            "speculation, and surface contradictions between sources rather "
            "than papering over them. Later you also act as the crew's "
            "fact-checker, verifying that the draft's claims are supported."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=verbose,
        max_iter=12,
    )

    report_writer = Agent(
        role="Report Writer",
        goal=(
            "Produce a clean, well-structured markdown report: an executive "
            "summary, one section per sub-question, and a sources list with "
            "every URL cited inline."
        ),
        backstory=(
            "You are an experienced technical writer. You write plainly and "
            "precisely, cite sources inline as you make claims, and never "
            "assert anything the analysis does not support. You produce "
            "portfolio-grade documents."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=verbose,
        max_iter=12,
    )

    return CrewAgents(
        research_lead=research_lead,
        web_researcher=web_researcher,
        analyst=analyst,
        report_writer=report_writer,
    )
