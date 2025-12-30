import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
- **get_course_outline**: Use for questions about course structure, lesson lists, what topics a course covers, or course overview
- **search_course_content**: Use for questions about specific course content or detailed educational materials

Tool Usage:
- Use get_course_outline when users ask about: course outline, lesson list, course structure, what lessons are in a course, course overview
- Use search_course_content when users ask about: specific topics, detailed explanations, content within lessons
- You may call tools sequentially (up to 2 times) when needed for:
  - Comparing information across multiple courses
  - Multi-part questions requiring different searches
  - When initial results inform a follow-up search
- When you have sufficient information, provide your answer directly
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results"
- **For course outlines**: Present the lesson list with each lesson on its own line using markdown list format (- Lesson N: Title)

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Maximum number of sequential tool calling rounds per query
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._execute_tool_loop(response, api_params, tool_manager, tools)
        
        # Return direct response
        return response.content[0].text
    
    def _execute_tool_loop(self, initial_response, base_params: Dict[str, Any],
                           tool_manager, tools: List) -> str:
        """
        Execute tool calls in a loop, allowing up to MAX_TOOL_ROUNDS sequential rounds.

        Terminates when:
        - Round limit reached (MAX_TOOL_ROUNDS)
        - Claude's response has no tool_use blocks (stop_reason != "tool_use")
        - Tool execution fails critically

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            tools: Tool definitions to include in subsequent API calls

        Returns:
            Final response text after tool execution
        """
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0

        while round_count < self.MAX_TOOL_ROUNDS:
            # Add assistant's tool use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                    except Exception as e:
                        tool_result = f"Error executing {content_block.name}: {str(e)}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

            # Add tool results to messages
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            round_count += 1

            # Prepare next API call - include tools if under limit
            if round_count < self.MAX_TOOL_ROUNDS:
                next_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"],
                    "tools": tools,
                    "tool_choice": {"type": "auto"}
                }
            else:
                # Final round - no tools, force answer
                next_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"]
                }

            # Make API call
            current_response = self.client.messages.create(**next_params)

            # Check if Claude wants more tools
            if current_response.stop_reason != "tool_use":
                break

        return current_response.content[0].text