import os
import logging
from typing import List, Dict, Any, Union
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from search_service import SearchService

# Import LLM clients with proper error handling
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
    def __init__(self, model: str = "gemini-2.0-flash-exp"):
        self.model = model
        self.search_service = SearchService()
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize LLM clients
        self._setup_llm_clients()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _setup_llm_clients(self):
        """Initialize LLM clients with API keys from environment"""
        try:
            # Gemini setup
            if GENAI_AVAILABLE:
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if gemini_api_key:
                    genai.configure(api_key=gemini_api_key)
            
            # OpenAI setup
            if OPENAI_AVAILABLE:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            
            # Anthropic setup
            if ANTHROPIC_AVAILABLE:
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                if anthropic_api_key:
                    self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
                
        except Exception as e:
            logging.error(f"Error setting up LLM clients: {e}")
    
    def _build_graph(self):
        """Build the langGraph workflow"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("search", self._search_node)
        workflow.add_node("generate", self._generate_node)
        
        # Add edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _extract_message_content(self, message):
        """Extract content from various message formats"""
        if isinstance(message, dict):
            return message.get("content", str(message))
        elif hasattr(message, 'content'):
            return message.content
        else:
            return str(message)
    
    async def _search_node(self, state: ChatState) -> ChatState:
        """Node for performing internet search"""
        try:
            # Get the latest message content
            if state["messages"]:
                latest_message = self._extract_message_content(state["messages"][-1])
            else:
                latest_message = ""
            
            # Determine if search is needed
            search_keywords = ["current", "latest", "news", "today", "recent", "2024", "2025", "what's", "happening"]
            needs_search = any(keyword in latest_message.lower() for keyword in search_keywords)
            
            if needs_search:
                search_results = await self.search_service.search(latest_message)
                state["search_results"] = search_results
                logging.info(f"Search performed for: {latest_message}")
            else:
                state["search_results"] = ""
                logging.info("No search needed for this query")
                
        except Exception as e:
            logging.error(f"Error in search node: {e}")
            state["search_results"] = ""
        
        return state
    
    async def _generate_node(self, state: ChatState) -> ChatState:
        """Node for generating LLM response"""
        try:
            messages = state["messages"]
            search_results = state.get("search_results", "")
            
            # Build context with search results if available
            context = ""
            if search_results:
                context = f"\n\nRelevant search results:\n{search_results}"
            
            # Get the latest user message
            if messages:
                user_message = self._extract_message_content(messages[-1])
            else:
                user_message = ""
                
            full_prompt = f"{user_message}{context}"
            
            # Convert messages to proper format for LLM call
            formatted_history = []
            if len(messages) > 1:
                for i, msg in enumerate(messages[:-1]):
                    content = self._extract_message_content(msg)
                    # Alternate between user and assistant based on position
                    role = "user" if i % 2 == 0 else "assistant"
                    formatted_history.append({"role": role, "content": content})
            
            # Generate response based on selected model
            response = await self._call_llm(full_prompt, formatted_history)
            
            state["final_response"] = response
            
        except Exception as e:
            logging.error(f"Error in generate node: {e}")
            state["final_response"] = "I apologize, but I encountered an error while processing your request. Please try again."
        
        return state
    
    async def _call_llm(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        """Call the appropriate LLM based on the selected model"""
        try:
            if self.model.startswith("gemini"):
                return await self._call_gemini(prompt, chat_history)
            elif self.model.startswith("gpt"):
                return await self._call_openai(prompt, chat_history)
            elif self.model.startswith("claude"):
                return await self._call_anthropic(prompt, chat_history)
            else:
                return await self._call_gemini(prompt, chat_history)  # Default fallback
                
        except Exception as e:
            logging.error(f"Error calling LLM {self.model}: {e}")
            return "I apologize, but I'm having trouble connecting to the AI service. Please check your API configuration and try again."
    
    async def _call_gemini(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        """Call Gemini API"""
        try:
            if not GENAI_AVAILABLE:
                raise Exception("Gemini API not available")
            
            # Use the selected Gemini model if provided
            model_name = self.model if self.model.startswith("gemini") else "gemini-2.0-flash-exp"
            model = genai.GenerativeModel(model_name)
            
            # Build conversation history
            conversation = []
            for exchange in chat_history:
                if exchange['role'] == 'user':
                    conversation.append(f"Human: {exchange['content']}")
                else:
                    conversation.append(f"Assistant: {exchange['content']}")
            
            # Add current prompt
            conversation.append(f"Human: {prompt}")
            conversation_text = "\n".join(conversation)
            
            response = await model.generate_content_async(conversation_text)
            return response.text
            
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            raise
    
    async def _call_openai(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        """Call OpenAI API"""
        try:
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")
            
            messages = [{"role": "system", "content": "You are a helpful AI assistant with access to current information through search results."}]
            
            # Add chat history
            for exchange in chat_history:
                messages.append({"role": exchange["role"], "content": exchange["content"]})
            
            # Add current message
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
        """Call Anthropic API"""
        try:
            if not self.anthropic_client:
                raise Exception("Anthropic client not initialized")
            
            messages = []
            
            # Add chat history
            for exchange in chat_history:
                messages.append({"role": exchange["role"], "content": exchange["content"]})
            
            # Add current message
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
        """Process a user message through the langGraph workflow"""
        try:
            # Convert chat history to simple message format
            messages = []
            for exchange in chat_history:
                if 'user' in exchange:
                    messages.append({"role": "user", "content": exchange['user']})
                if 'bot' in exchange:
                    messages.append({"role": "assistant", "content": exchange['bot']})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Initialize state
            initial_state = {
                "messages": messages,
                "search_results": "",
                "final_response": ""
            }
            
            # Run through the graph
            result = await self.graph.ainvoke(initial_state)
            
            return result["final_response"]
            
        except Exception as e:
            logging.error(f"Error processing message through graph: {e}")
            return "I apologize, but I encountered an error while processing your message. Please try again."
