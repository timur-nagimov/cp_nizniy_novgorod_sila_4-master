from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore

from typing import Any, Dict, List, Optional

import prompts


class ResponseWithChatHistory(CustomQueryComponent):
    llm: LangChainLLM = Field(..., description="LLM")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use for the LLM"
    )
    context_prompt: str = Field(
        default=prompts.DEFAULT_CONTEXT_PROMPT,
        description="Context prompt to use for the LLM",
    )

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"nodes", "query_str"}

    @property
    def _output_keys(self) -> set:
        return {"response", "context"}

    def _prepare_context(
        self,
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[ChatMessage]:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_text()
            node_meta = node.dict()["node"]["metadata"]

            print(node_meta)
            print(node.id_)
            
            if node_meta.get('level_3') == '0 ':
                node_context += f"### {node_meta.get('level_2')} \n #### Страница {int(node_meta.get('page'))}  \n {node_text} \n\n РИСУНКИ: {node_meta.get('figures')} \n\n\n"
            else:
                node_context += f"### {node_meta.get('level_3')} \n #### Страница {int(node_meta.get('page'))}  \n {node_text} \n\n РИСУНКИ: {node_meta.get('figures')} \n\n\n"

        formatted_context = self.context_prompt.format(
            node_context=node_context, query_str=query_str
        )

        return formatted_context, node_context

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context, node_context = self._prepare_context(nodes, query_str)

        response = self.llm.complete(prepared_context)

        return {"response": response, "context": node_context}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the component asynchronously."""
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context, node_context = self._prepare_context(nodes, query_str)

        response = await self.llm.acomplete(prepared_context)

        return {"response": response, "context": node_context}
