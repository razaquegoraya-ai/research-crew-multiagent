"""The task graph the crew executes, in order.

The pipeline is deliberately *sequential* rather than hierarchical: each task's
output is passed as context to the next, which produces a clean, deterministic
trace of hand-offs — exactly the visible orchestration this project is meant to
showcase.

Flow::

    plan → research → analyze → draft → review → finalize
     (Lead) (Researcher) (Analyst) (Writer) (Analyst)  (Writer)

The ``review`` step is the lightweight fact-check: the Analyst inspects the
Writer's draft for unsupported claims before the Writer produces the final,
corrected report.
"""

from __future__ import annotations

from crewai import Task

from research_crew.agents.crew_agents import CrewAgents
from research_crew.models.schemas import Depth


def build_tasks(agents: CrewAgents, topic: str, depth: Depth) -> list[Task]:
    """Build the ordered list of tasks for a single research run.

    Args:
        agents: The assembled crew agents.
        topic: The research topic or company.
        depth: Requested depth, which tunes sub-question count and thoroughness.

    Returns:
        Tasks in execution order. The final task's output is the report.
    """
    n = depth.sub_question_count
    per_q = depth.results_per_question

    plan_task = Task(
        name="plan_sub_questions",
        description=(
            f"The research topic is:\n\n\"{topic}\"\n\n"
            f"Break this into exactly {n} focused, non-overlapping "
            "sub-questions that together fully cover the topic. Each "
            "sub-question must be answerable with evidence from public web "
            "sources. Avoid vague or opinion-only questions."
        ),
        expected_output=(
            f"A numbered list of {n} sub-questions, each on its own line, with "
            "a one-sentence note on why it matters. No preamble."
        ),
        agent=agents.research_lead,
    )

    research_task = Task(
        name="gather_sources",
        description=(
            "Using the sub-questions from the Research Lead, use the "
            "web_search tool to investigate EACH sub-question (aim for about "
            f"{per_q} quality sources per question). For every finding, record "
            "the exact source URL. Prefer recent, credible sources. If a "
            "search returns nothing useful, say so explicitly and move on — do "
            "not fabricate results."
        ),
        expected_output=(
            "For each sub-question: a short heading, then 3-6 bullet findings, "
            "each finding followed by its source URL in parentheses. Group "
            "findings under the sub-question they answer."
        ),
        agent=agents.web_researcher,
        context=[plan_task],
    )

    analysis_task = Task(
        name="synthesize_insights",
        description=(
            "Synthesize the researcher's raw findings into structured "
            "insight. For each sub-question, state the best-supported "
            "conclusions and the evidence behind them. Explicitly flag any "
            "points where sources CONFLICT or where evidence is weak. Remove "
            "filler and repetition. Preserve the source URLs so they can be "
            "cited later."
        ),
        expected_output=(
            "A structured synthesis organized by sub-question, with a clearly "
            "labeled 'Conflicts & caveats' note wherever sources disagree or "
            "evidence is thin. Keep URLs attached to their claims."
        ),
        agent=agents.analyst,
        context=[plan_task, research_task],
    )

    draft_task = Task(
        name="write_draft_report",
        description=(
            f"Write a polished markdown report on \"{topic}\" using the "
            "analysis. Structure:\n"
            "1. A top H1 title.\n"
            "2. An 'Executive Summary' section (3-6 sentences).\n"
            "3. One '##' section per sub-question, answering it with inline "
            "citations like ([source](URL)).\n"
            "4. A final '## Sources' section listing every URL used.\n"
            "Every substantive claim must carry an inline citation. Do not "
            "invent facts or URLs not present in the analysis."
        ),
        expected_output=(
            "A complete markdown report with title, executive summary, one "
            "section per sub-question with inline citations, and a Sources "
            "list. Markdown only — no surrounding commentary."
        ),
        agent=agents.report_writer,
        context=[plan_task, research_task, analysis_task],
    )

    review_task = Task(
        name="fact_check_review",
        description=(
            "Act as fact-checker. Review the Writer's draft against the "
            "research findings and analysis. Identify every claim that lacks a "
            "supporting source or that overstates what the sources show. "
            "Produce a concise, actionable list of issues. If the draft is "
            "fully supported, say so explicitly."
        ),
        expected_output=(
            "A bulleted 'Review notes' list. Each item names the specific "
            "claim and states the problem (e.g. 'no source', 'overstated', "
            "'contradicts source X'), or a single line confirming the draft is "
            "fully supported."
        ),
        agent=agents.analyst,
        context=[research_task, analysis_task, draft_task],
    )

    finalize_task = Task(
        name="finalize_report",
        description=(
            "Produce the FINAL markdown report by revising your draft to "
            "address every item in the Analyst's review notes: remove or "
            "soften unsupported claims, add missing citations where the "
            "sources support them, and ensure the Sources section is complete "
            "and accurate. The output must be the finished report only."
        ),
        expected_output=(
            "The final, corrected markdown report: title, executive summary, "
            "one section per sub-question with inline citations, and a "
            "complete Sources list. Markdown only — no commentary, no review "
            "notes."
        ),
        agent=agents.report_writer,
        context=[analysis_task, draft_task, review_task],
    )

    return [
        plan_task,
        research_task,
        analysis_task,
        draft_task,
        review_task,
        finalize_task,
    ]
