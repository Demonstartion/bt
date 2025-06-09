import os
import logging
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from search_service import SearchService

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

class ChatState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    search_results: str
    final_response: str

class ChatGraph:
    def __init__(self, model: str = "gemini-2.0-flash-exp", company: str = "the company", persona: str = ""):
        self.model = model
        self.company = company
        self.persona = persona
        self.search_service = SearchService()
        self.openai_client = None
        self.anthropic_client = None
        self._setup_llm_clients()
        self.graph = self._build_graph()
    
    def _setup_llm_clients(self):
        try:
            if GENAI_AVAILABLE:
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if gemini_api_key:
                    genai.configure(api_key=gemini_api_key)
            if OPENAI_AVAILABLE:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            if ANTHROPIC_AVAILABLE:
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                if anthropic_api_key:
                    self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        except Exception as e:
            logging.error(f"Error setting up LLM clients: {e}")

    def _build_graph(self):
        workflow = StateGraph(ChatState)
        workflow.add_node("should_search", self._should_search_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("generate", self._generate_node)
        workflow.set_entry_point("should_search")
        workflow.add_conditional_edges(
            "should_search",
            lambda state: state.get("should_search", False),
            {
                True: "search",
                False: "generate"
            }
        )
        workflow.add_edge("search", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def _extract_message_content(self, message):
        if isinstance(message, dict):
            return message.get("content", str(message))
        elif hasattr(message, 'content'):
            return message.content
        else:
            return str(message)

    async def _should_search_node(self, state: ChatState) -> ChatState:
        try:
            messages = state["messages"]
            user_message = self._extract_message_content(messages[-1]) if messages else ""
            context = (
                f"{self.persona}\n"
                f"Decide if you need to perform a web search to answer the following user question. "
                f"If you need up-to-date or external information, reply ONLY with YES. Otherwise, reply ONLY with NO.\n"
                f"User question: {user_message}"
            )
            decision = await self._call_llm(context, [])
            state["should_search"] = decision.strip().upper().startswith("Y")
        except Exception as e:
            logging.error(f"Error in should_search_node: {e}")
            state["should_search"] = False
        return state

    async def _search_node(self, state: ChatState) -> ChatState:
        try:
            if state["messages"]:
                latest_message = self._extract_message_content(state["messages"][-1])
            else:
                latest_message = ""
            search_results = await self.search_service.search(latest_message)
            state["search_results"] = search_results
            logging.info(f"Search performed for: {latest_message}")
        except Exception as e:
            logging.error(f"Error in search node: {e}")
            state["search_results"] = ""
        return state

    async def _generate_node(self, state: ChatState) -> ChatState:
        try:
            messages = state["messages"]
            search_results = state.get("search_results", "")
            user_message = self._extract_message_content(messages[-1]) if messages else ""
            base_context = (
                f"{self.persona} "
                f"If search results are provided, prefer using them to provide the most accurate and recent information.\n"
            )
            if search_results:
                base_context += f"\nRelevant recent web search results:\n{search_results}\n"
            base_context += f"\nUser: {user_message}"
            formatted_history = []
            for i, msg in enumerate(messages[:-1]):
                content = self._extract_message_content(msg)
                role = "user" if i % 2 == 0 else "assistant"
                formatted_history.append({"role": role, "content": content})
            response = await self._call_llm(base_context, formatted_history)
            state["final_response"] = response
        except Exception as e:
            logging.error(f"Error in generate node: {e}")
            state["final_response"] = "I apologize, but I encountered an error while processing your request. Please try again."
        return state

    async def _call_llm(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        try:
            if self.model.startswith("gemini"):
                return await self._call_gemini(prompt, chat_history)
            elif self.model.startswith("gpt"):
                return await self._call_openai(prompt, chat_history)
            elif self.model.startswith("claude"):
                return await self._call_anthropic(prompt, chat_history)
            else:
                return await self._call_gemini(prompt, chat_history)
        except Exception as e:
            logging.error(f"Error calling LLM {self.model}: {e}")
            return "I apologize, but I'm having trouble connecting to the AI service. Please check your API configuration and try again."

    async def _call_gemini(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        try:
            if not GENAI_AVAILABLE:
                raise Exception("Gemini API not available")
            model_name = self.model if self.model.startswith("gemini") else "gemini-2.0-flash-exp"
            model = genai.GenerativeModel(model_name)
            conversation = []
            for exchange in chat_history:
                if exchange['role'] == 'user':
                    conversation.append(f"Human: {exchange['content']}")
                else:
                    conversation.append(f"Assistant: {exchange['content']}")
            conversation.append(f"Human: {prompt}")
            conversation_text = "\n".join(conversation)
            response = await model.generate_content_async(conversation_text)
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            raise

    async def _call_openai(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        try:
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")
            messages = [{"role": "system", "content": self.persona or "You are a helpful customer service AI."}]
            for exchange in chat_history:
                messages.append({"role": exchange["role"], "content": exchange["content"]})
            messages.append({"role": "user", "content": prompt})
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content or "No response generated"
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise

    async def _call_anthropic(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        try:
            if not self.anthropic_client:
                raise Exception("Anthropic client not initialized")
            messages = []
            for exchange in chat_history:
                messages.append({"role": exchange["role"], "content": exchange["content"]})
            messages.append({"role": "user", "content": prompt})
            response = await self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            raise

    async def process_message(self, message: str, chat_history: List[Dict[str, str]]) -> str:
        messages = []
        for exchange in chat_history:
            if 'user' in exchange:
                messages.append({"role": "user", "content": exchange['user']})
            if 'bot' in exchange:
                messages.append({"role": "assistant", "content": exchange['bot']})
        messages.append({"role": "user", "content": message})
        initial_state = {
            "messages": messages,
            "search_results": "",
            "final_response": ""
        }
        result = await self.graph.ainvoke(initial_state)
        return result["final_response"]

    async def stream_message(self, message: str, chat_history: List[Dict[str, str]]):
        # Streaming for OpenAI/Gemini only (Anthropic doesn't support streaming as of now)
        messages = []
        for exchange in chat_history:
            if 'user' in exchange:
                messages.append({"role": "user", "content": exchange['user']})
            if 'bot' in exchange:
                messages.append({"role": "assistant", "content": exchange['bot']})
        messages.append({"role": "user", "content": message})
        # Decide if search is needed
        state = {"messages": messages, "search_results": "", "final_response": ""}
        state = await self._should_search_node(state)
        if state.get("should_search"):
            state = await self._search_node(state)
        # Now stream the generation
        search_results = state.get("search_results", "")
        user_message = self._extract_message_content(messages[-1])
        persona_context = f"{self.persona} " if self.persona else ""
        base_context = (
            f"{persona_context}If search results are provided, prefer using them to provide the most accurate and recent information.\n"
        )
        if search_results:
            base_context += f"\nRelevant recent web search results:\n{search_results}\n"
        base_context += f"\nUser: {user_message}"
        formatted_history = []
        for i, msg in enumerate(messages[:-1]):
            content = self._extract_message_content(msg)
            role = "user" if i % 2 == 0 else "assistant"
            formatted_history.append({"role": role, "content": content})
        if self.model.startswith("gpt") and self.openai_client:
            # Stream OpenAI
            async for chunk in self._stream_openai(base_context, formatted_history):
                yield chunk
        elif self.model.startswith("gemini") and GENAI_AVAILABLE:
            async for chunk in self._stream_gemini(base_context, formatted_history):
                yield chunk
        else:
            # Fallback: not streaming, just yield full response
            response = await self._call_llm(base_context, formatted_history)
            async for chunk in response:
                yield chunk

    def _stream_openai(self, prompt, chat_history):
        # OpenAI streaming (sync generator, since Quart expects async, wrap in async if needed)
        import asyncio
        async def inner():
            messages = [{"role": "system", "content": self.persona or "You are a helpful customer service AI."}]
            for exchange in chat_history:
                messages.append({"role": exchange["role"], "content": exchange["content"]})
            messages.append({"role": "user", "content": prompt})
            stream = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )
            async for chunk in stream:
                part = chunk.choices[0].delta.content or ""
                if part:
                    yield part
        return inner()

    async def _stream_gemini(self, prompt, chat_history):
        if not GENAI_AVAILABLE:
            yield "Sorry, Gemini API not configured."
            return
        model_name = self.model if self.model.startswith("gemini") else "gemini-2.0-flash-exp"
        model = genai.GenerativeModel(model_name)
        conversation = []
        for exchange in chat_history:
            if exchange['role'] == 'user':
                conversation.append(f"Human: {exchange['content']}")
            else:
                conversation.append(f"Assistant: {exchange['content']}")
        conversation.append(f"Human: {prompt}")
        conversation_text = "\n".join(conversation)
        response = await model.generate_content_async(conversation_text, stream=True)
        async for chunk in response:
            if hasattr(chunk, "text"):
                yield chunk.text
