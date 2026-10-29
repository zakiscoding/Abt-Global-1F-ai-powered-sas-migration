"""LangGraph workflow: document -> save markdown under docs/migration/."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import config, prompts
from .collect import collect

_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


class DocumentState(TypedDict, total=False):
    program_id: str
    documentation: str
    output_path: str


def build_llm() -> BaseChatModel:
    # Long SAS + macros prompts can take a while; don't time out early.
    return init_chat_model(config.MODEL, timeout=900)


def _ask(llm: BaseChatModel, prompt: str) -> str:
    response = llm.invoke([SystemMessage(prompts.SYSTEM), HumanMessage(prompt)])
    finish_reason = response.response_metadata.get("finish_reason")
    if finish_reason in _BLOCKED:
        raise RuntimeError(f"Model blocked the response (finish_reason={finish_reason})")
    if finish_reason == "MAX_TOKENS":
        raise RuntimeError("Model hit its output token limit; the response is truncated.")
    return response.text


def build_graph(llm: BaseChatModel | None = None):
    llm = llm or build_llm()

    def document(state: DocumentState) -> dict:
        ctx = collect(state["program_id"])
        prompt = prompts.fill(
            prompts.DOCUMENT,
            mapping=ctx.mapping,
            outputs=", ".join(ctx.outputs),
            macros=ctx.macros,
            sas_file=ctx.sas_file,
            sas_code=ctx.sas_code,
        )
        documentation = _ask(llm, prompt)
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.DOCS_DIR / f"{ctx.program_id}.md"
        path.write_text(documentation.strip() + "\n", encoding="utf-8")
        return {"documentation": documentation, "output_path": str(path)}

    graph = StateGraph(DocumentState)
    graph.add_node("document", document)
    graph.add_edge(START, "document")
    graph.add_edge("document", END)
    return graph.compile()


def document_program(program_id: str, llm: BaseChatModel | None = None) -> Path:
    result = build_graph(llm).invoke({"program_id": program_id})
    return Path(result["output_path"])
